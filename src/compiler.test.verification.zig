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
const verifyTextWithoutDegradationWithSummaryInlineDepth = h.verifyTextWithoutDegradationWithSummaryInlineDepth;
const verifyPackageWithoutDegradation = h.verifyPackageWithoutDegradation;
const verifyPackageWithoutDegradationWithImportedSummaryMode = h.verifyPackageWithoutDegradationWithImportedSummaryMode;
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

test "verification keeps public parameter refinement guards for untrusted public input" {
    const source_text =
        \\contract C {
        \\    pub fn accept(
        \\        min_value: MinValue<u256, 1>,
        \\        max_value: MaxValue<u256, 100>,
        \\        in_range: InRange<u256, 1, 100>,
        \\        basis_points: BasisPoints<u256>,
        \\        non_zero: NonZero<u256>,
        \\        target: NonZeroAddress,
        \\    ) -> bool {
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "accept");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 6), result.diagnostics_len);
    try testing.expectEqualStrings("", result.error_kinds);
}

test "verification applies ensures_ok and ensures_err only to matching error-union exits" {
    const source_text =
        \\error Rejected;
        \\
        \\contract C {
        \\    pub fn choose(flag: bool, value: u256) -> !u256 | Rejected
        \\        ensures_ok(result == value)
        \\        ensures_err(value == value)
        \\    {
        \\        if (!flag) {
        \\            return Rejected;
        \\        }
        \\        return value;
        \\    }
        \\
        \\    pub fn fail_only() -> !bool | Rejected
        \\        ensures_ok(false)
        \\        ensures_err(true)
        \\    {
        \\        return Rejected;
        \\    }
        \\
        \\    pub fn ok_only() -> !bool | Rejected
        \\        ensures_ok(result == true)
        \\        ensures_err(false)
        \\    {
        \\        return true;
        \\    }
        \\}
    ;

    var choose = try verifyTextWithoutDegradation(source_text, "choose");
    defer choose.deinit(testing.allocator);
    try testing.expect(choose.success);
    try testing.expect(!choose.degraded);
    try testing.expectEqual(@as(usize, 0), choose.errors_len);

    var fail_only = try verifyTextWithoutDegradation(source_text, "fail_only");
    defer fail_only.deinit(testing.allocator);
    try testing.expect(fail_only.success);
    try testing.expect(!fail_only.degraded);
    try testing.expectEqual(@as(usize, 0), fail_only.errors_len);

    var ok_only = try verifyTextWithoutDegradation(source_text, "ok_only");
    defer ok_only.deinit(testing.allocator);
    try testing.expect(ok_only.success);
    try testing.expect(!ok_only.degraded);
    try testing.expectEqual(@as(usize, 0), ok_only.errors_len);
}

test "verification reports failing ensures_ok and ensures_err on matching exits" {
    const source_text =
        \\error Rejected;
        \\
        \\contract C {
        \\    pub fn bad_ok() -> !bool | Rejected
        \\        ensures_ok(false)
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn bad_err() -> !bool | Rejected
        \\        ensures_err(false)
        \\    {
        \\        return Rejected;
        \\    }
        \\}
    ;

    var bad_ok = try verifyTextWithoutDegradation(source_text, "bad_ok");
    defer bad_ok.deinit(testing.allocator);
    try testing.expect(bad_ok.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", bad_ok.error_kinds);

    var bad_err = try verifyTextWithoutDegradation(source_text, "bad_err");
    defer bad_err.deinit(testing.allocator);
    try testing.expect(bad_err.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", bad_err.error_kinds);
}

test "verification loop invariant step excludes break exit paths" {
    const source_text =
        \\contract C {
        \\    pub fn f() -> u256 {
        \\        var counter: u256 = 0;
        \\        while (true)
        \\            invariant counter <= 6
        \\        {
        \\            counter = counter + 1;
        \\            if (counter > 5) {
        \\                break;
        \\            }
        \\        }
        \\        return counter;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.error_kinds);
}

test "verification proves goal-position forall postconditions with skolem counterexamples" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract C {
        \\    pub fn bounded_by_input(x: u256, k: u256) -> bool
        \\        requires k < std.constants.U256_MAX / 2
        \\        requires x < std.constants.U256_MAX - k
        \\        ensures (forall i: u256 where i < x => i <= x + k)
        \\    {
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "bounded_by_input");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.error_kinds);
}

test "verification reports false goal-position forall postconditions with skolem readable witness" {
    const source_text =
        \\contract C {
        \\    pub fn quantified_false(x: u256) -> bool
        \\        requires x > 0
        \\        ensures (forall i: u256 where i < x => i >= x)
        \\    {
        \\        return true;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.filter_function_name = "quantified_false";

    var result = try verifier.runVerificationPassPreparedSequential(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors.items.len);
    const err = &result.errors.items[0];
    try testing.expectEqualStrings("PostconditionViolation", @tagName(err.error_type));

    var found_i_witness = false;
    if (err.counterexample) |*ce| {
        var iter = ce.variables.iterator();
        while (iter.next()) |entry| {
            if (std.mem.indexOf(u8, entry.key_ptr.*, "$ora.goal.skolem.i.") != null) {
                found_i_witness = true;
                break;
            }
        }
    }
    try testing.expect(found_i_witness);

    var artifacts = try verifier.buildSmtReport(hir_result.module.raw_module, "/tmp/quantified_false.ora", &result);
    defer artifacts.deinit(testing.allocator);
    try testing.expect(std.mem.indexOf(u8, artifacts.json, "\"unknown\":0") != null);
    try testing.expect(std.mem.indexOf(u8, artifacts.json, "\"status\":\"UNKNOWN\"") == null);
    try testing.expect(std.mem.indexOf(u8, artifacts.markdown, "assumptions inconsistent") == null);
    try testing.expect(std.mem.indexOf(u8, artifacts.json, "\"vacuous\":true") == null);
}

test "verification supports enum constants in stored struct fields without degradation" {
    const source_text =
        \\enum Status { Pending, Filled }
        \\enum ErrorCode: string { InvalidInput, Unauthorized }
        \\
        \\struct Order {
        \\    id: u256,
        \\    amount: u256,
        \\    status: Status,
        \\}
        \\
        \\contract C {
        \\    storage var orders: map<u256, Order>;
        \\    storage var last_error: ErrorCode;
        \\
        \\    pub fn create(id: u256, amount: u256) -> u256 {
        \\        let order: Order = Order {
        \\            id: id,
        \\            amount: amount,
        \\            status: Status.Pending,
        \\        };
        \\        orders[id] = order;
        \\        last_error = ErrorCode.InvalidInput;
        \\        return amount;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "create");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports explicit string enum storage roundtrip without degradation" {
    const source_text =
        \\enum ErrorCode : string {
        \\    InvalidInput = "ERR_INVALID_INPUT",
        \\    Unauthorized = "ERR_UNAUTHORIZED",
        \\}
        \\
        \\contract C {
        \\    storage var last_error: ErrorCode;
        \\
        \\    pub fn remember() -> bool
        \\        ensures last_error == ErrorCode.InvalidInput
        \\    {
        \\        last_error = ErrorCode.InvalidInput;
        \\        return last_error == ErrorCode.InvalidInput;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "remember");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
}

test "verification supports explicit bytes enum storage roundtrip without degradation" {
    const source_text =
        \\enum Signature : bytes {
        \\    A = hex"deadbeef",
        \\    B = hex"c0ffee",
        \\}
        \\
        \\contract C {
        \\    storage var sig: Signature;
        \\
        \\    pub fn remember() -> bool
        \\        ensures sig == Signature.B
        \\    {
        \\        sig = Signature.B;
        \\        return sig == Signature.B;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "remember");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
}

test "verification accepts wrapping shifts with narrow shift operand without degradation" {
    const source_text =
        \\contract C {
        \\    pub fn shift_left(a: u256, bits: u8) -> u256 {
        \\        let shifted: u256 = a <<% bits;
        \\        return 1 +% shifted;
        \\    }
        \\
        \\    pub fn shift_right(a: i256, bits: u8) -> i256 {
        \\        return a >>% bits;
        \\    }
        \\}
    ;

    var left_result = try verifyTextWithoutDegradation(source_text, "shift_left");
    defer left_result.deinit(testing.allocator);
    try testing.expect(left_result.success);
    try testing.expectEqual(@as(usize, 0), left_result.errors_len);
    try testing.expect(!left_result.degraded);

    var right_result = try verifyTextWithoutDegradation(source_text, "shift_right");
    defer right_result.deinit(testing.allocator);
    try testing.expect(right_result.success);
    try testing.expectEqual(@as(usize, 0), right_result.errors_len);
    try testing.expect(!right_result.degraded);
}

test "verification accepts lowered EVM environment builtins without degradation" {
    const source_text =
        \\contract EnvProbe {
        \\    pub fn sender() -> address {
        \\        return std.msg.sender();
        \\    }
        \\
        \\    pub fn origin() -> address {
        \\        return std.tx.origin();
        \\    }
        \\
        \\    pub fn coinbase() -> address {
        \\        return std.block.coinbase();
        \\    }
        \\
        \\    pub fn numeric_env() -> u256 {
        \\        return std.msg.value() +% std.transaction.gasprice() +%
        \\            std.block.timestamp() +% std.block.number();
        \\    }
        \\}
    ;

    inline for (.{ "sender", "origin", "coinbase", "numeric_env" }) |function_name| {
        var result = try verifyTextWithoutDegradation(source_text, function_name);
        defer result.deinit(testing.allocator);
        try testing.expect(result.success);
        try testing.expectEqual(@as(usize, 0), result.errors_len);
        try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
        try testing.expect(!result.degraded);
        try testing.expectEqualStrings("", result.soundness_losses);
    }
}

test "verification accepts runtime keccak256 without degradation" {
    const source_text =
        \\pub fn hash_identity(data: bytes) -> u256
        \\    ensures(result == @keccak256(data))
        \\{
        \\    return @keccak256(data);
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "hash_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification models computed storage word store then load" {
    const source_text =
        \\contract C {
        \\    pub fn roundtrip(owner: address, value: u256) -> u256
        \\        modifies @storageRange(@storageDerive("verify.computed.roundtrip", owner), 1)
        \\        ensures result == value
        \\    {
        \\        let slot: StorageSlot = @storageDerive("verify.computed.roundtrip", owner);
        \\        @storageWordStore(slot, 0, value);
        \\        return @storageWordLoad(slot, 0);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "roundtrip");
    defer result.deinit(testing.allocator);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.error_kinds);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(result.success);
}

test "verification rejects false postcondition over computed storage" {
    const source_text =
        \\contract C {
        \\    pub fn false_roundtrip(owner: address, value: u256) -> u256
        \\        modifies @storageRange(@storageDerive("verify.computed.false.roundtrip", owner), 1)
        \\        ensures result != value
        \\    {
        \\        let slot: StorageSlot = @storageDerive("verify.computed.false.roundtrip", owner);
        \\        @storageWordStore(slot, 0, value);
        \\        return @storageWordLoad(slot, 0);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "false_roundtrip");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", result.error_kinds);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification models resource create as a map storage update" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\contract C {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn mint(owner: address, amount: TokenUnit)
        \\        modifies balances[owner]
        \\        requires balances[owner] <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\        ensures balances[owner] == old(balances[owner]) + amount
        \\    {
        \\        @create(balances[owner], amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "mint");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rejects false frame claim after resource create" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\contract C {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn mint(owner: address, amount: TokenUnit)
        \\        modifies balances[owner]
        \\        requires amount > 0
        \\        requires balances[owner] <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\        ensures balances[owner] == old(balances[owner])
        \\    {
        \\        @create(balances[owner], amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "mint");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", result.error_kinds);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification models resource create as a direct storage update" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\contract C {
        \\    storage var reserve: Resource<TokenUnit>;
        \\
        \\    pub fn mint(amount: TokenUnit)
        \\        modifies reserve
        \\        requires reserve <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\        ensures reserve == old(reserve) + amount
        \\    {
        \\        @create(reserve, amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "mint");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification models resource create as a direct transient update" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\contract C {
        \\    tstore var scratch: Resource<TokenUnit>;
        \\
        \\    pub fn mint(amount: TokenUnit)
        \\        requires scratch <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\        ensures scratch == old(scratch) + amount
        \\    {
        \\        @create(scratch, amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "mint");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rejects false frame claim after direct transient resource create" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\contract C {
        \\    tstore var scratch: Resource<TokenUnit>;
        \\
        \\    pub fn mint(amount: TokenUnit)
        \\        requires amount > 0
        \\        requires scratch <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\        ensures scratch == old(scratch)
        \\    {
        \\        @create(scratch, amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "mint");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", result.error_kinds);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification models resource move across distinct roots" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\contract C {
        \\    storage var pending: map<address, Resource<TokenUnit>>;
        \\    storage var settled: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn settle(from: address, to: address, amount: TokenUnit)
        \\        modifies pending[from], settled[to]
        \\        requires pending[from] >= amount
        \\        requires settled[to] <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\        ensures pending[from] == old(pending[from]) - amount
        \\        ensures settled[to] == old(settled[to]) + amount
        \\    {
        \\        @move(pending[from], settled[to], amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "settle");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification models signed resource moves with non-negative amounts" {
    const source_text =
        \\resource DebtUnit = i256;
        \\
        \\contract C {
        \\    storage var debts: map<address, Resource<DebtUnit>>;
        \\    storage var settled: Resource<DebtUnit>;
        \\
        \\    pub fn settle(from: address, amount: DebtUnit)
        \\        modifies debts[from], settled
        \\        requires amount >= 0
        \\        requires amount <= 100
        \\        requires debts[from] >= -100
        \\        requires settled <= 100
        \\        ensures debts[from] == old(debts[from]) - amount
        \\        ensures settled == old(settled) + amount
        \\    {
        \\        @move(debts[from], settled, amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "settle");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rejects signed resource create without non-negative amount proof" {
    const source_text =
        \\resource DebtUnit = i256;
        \\
        \\contract C {
        \\    storage var debts: map<address, Resource<DebtUnit>>;
        \\
        \\    pub fn mint(owner: address, amount: DebtUnit)
        \\        modifies debts[owner]
        \\        requires amount <= 100
        \\        requires debts[owner] <= 100
        \\        ensures debts[owner] == old(debts[owner]) + amount
        \\    {
        \\        @create(debts[owner], amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "mint");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rejects false frame claim after resource move" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\contract C {
        \\    storage var pending: map<address, Resource<TokenUnit>>;
        \\    storage var settled: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn settle(from: address, to: address, amount: TokenUnit)
        \\        modifies pending[from], settled[to]
        \\        requires amount > 0
        \\        requires pending[from] >= amount
        \\        requires settled[to] <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\        ensures pending[from] == old(pending[from])
        \\    {
        \\        @move(pending[from], settled[to], amount);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "settle");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", result.error_kinds);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification preserves computed storage slots returned by private helpers" {
    const source_text =
        \\contract C {
        \\    fn dataSlot(owner: address) -> StorageSlot {
        \\        return @storageDerive("verify.computed.private.slot", owner);
        \\    }
        \\
        \\    pub fn roundtrip(owner: address, value: u256) -> u256
        \\        modifies @storageRange(@storageDerive("verify.computed.private.slot", owner), 1)
        \\        ensures result == value
        \\    {
        \\        let slot: StorageSlot = dataSlot(owner);
        \\        @storageWordStore(slot, 0, value);
        \\        return @storageWordLoad(slot, 0);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "roundtrip");
    defer result.deinit(testing.allocator);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.error_kinds);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(result.success);
}

test "verification preserves computed storage slots through std storage helpers" {
    const source_text =
        \\comptime const std_storage = @import("std/storage");
        \\
        \\contract C {
        \\    fn dataSlot(owner: address) -> StorageSlot {
        \\        return @storageDerive("verify.computed.std.slot", owner);
        \\    }
        \\
        \\    pub fn roundtrip(owner: address, value: u256) -> u256
        \\        modifies @storageRange(@storageDerive("verify.computed.std.slot", owner), 1)
        \\        ensures result == value
        \\    {
        \\        let slot: StorageSlot = dataSlot(owner);
        \\        std_storage.words.store(slot, 0, value);
        \\        return std_storage.words.load(slot, 0);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "roundtrip");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification preserves computed storage std helper effects with modifies clauses" {
    const source_text =
        \\comptime const std_storage = @import("std/storage");
        \\
        \\contract C {
        \\    fn dataSlot(owner: address) -> StorageSlot {
        \\        return @storageDerive("verify.computed.std.no_modifies", owner);
        \\    }
        \\
        \\    pub fn write(owner: address, value: u256)
        \\        modifies std_storage.range(std_storage.derive("verify.computed.std.no_modifies", owner), 1)
        \\    {
        \\        let slot: StorageSlot = dataSlot(owner);
        \\        std_storage.words.store(slot, 0, value);
        \\    }
        \\
        \\    pub fn read(owner: address) -> u256 {
        \\        let slot: StorageSlot = dataSlot(owner);
        \\        return std_storage.words.load(slot, 0);
        \\    }
        \\}
    ;

    var write = try verifyTextWithoutDegradation(source_text, "write");
    defer write.deinit(testing.allocator);
    try testing.expect(write.success);
    try testing.expectEqual(@as(usize, 0), write.errors_len);
    try testing.expectEqual(@as(usize, 0), write.diagnostics_len);
    try testing.expect(!write.degraded);
    try testing.expectEqualStrings("", write.soundness_losses);

    var read = try verifyTextWithoutDegradation(source_text, "read");
    defer read.deinit(testing.allocator);
    try testing.expect(read.success);
    try testing.expectEqual(@as(usize, 0), read.errors_len);
    try testing.expectEqual(@as(usize, 0), read.diagnostics_len);
    try testing.expect(!read.degraded);
    try testing.expectEqualStrings("", read.soundness_losses);
}

test "verification preserves imported computed storage helpers through std storage calls" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "dep.ora",
        .data =
        \\comptime const std_storage = @import("std/storage");
        \\
        \\pub fn dataSlot(owner: address) -> StorageSlot {
        \\    return std_storage.derive("verify.computed.imported.std.slot", owner);
        \\}
        ,
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\comptime const std_storage = @import("std/storage");
        \\
        \\contract C {
        \\    pub fn write(owner: address, value: u256)
        \\        modifies std_storage.range(std_storage.derive("verify.computed.imported.std.slot", owner), 1)
        \\    {
        \\        std_storage.words.store(dep.dataSlot(owner), 0, value);
        \\    }
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var result = try verifyPackageWithoutDegradation(root_path, "write");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification preserves computed storage slots through std range erase helper" {
    const source_text =
        \\comptime const std_storage = @import("std/storage");
        \\
        \\contract C {
        \\    fn dataSlot(owner: address) -> StorageSlot {
        \\        return @storageDerive("verify.computed.std.erase", owner);
        \\    }
        \\
        \\    pub fn clear(owner: address) -> bool
        \\        modifies std_storage.range(std_storage.derive("verify.computed.std.erase", owner), 2)
        \\        ensures result
        \\    {
        \\        let slot: StorageSlot = dataSlot(owner);
        \\        std_storage.words.store(slot, 0, 1);
        \\        std_storage.words.store(slot, 1, 2);
        \\        let range: StorageRange = std_storage.range(slot, 2);
        \\        std_storage.words.erase(range);
        \\        return std_storage.words.load(slot, 0) == 0 && std_storage.words.load(slot, 1) == 0;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "clear");
    defer result.deinit(testing.allocator);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(result.success);
}

test "verification preserves computed storage slots through fixed size data helpers" {
    const source_text =
        \\comptime const fixed = @import("std/storage/fixed_size_data");
        \\
        \\contract C {
        \\    fn dataSlot(owner: address) -> StorageSlot {
        \\        return @storageDerive("verify.computed.fixed", owner);
        \\    }
        \\
        \\    pub fn roundtrip(owner: address, first: u256, second: u256) -> bool
        \\        modifies @storageRange(@storageDerive("verify.computed.fixed", owner), 2)
        \\        requires first != 0
        \\        ensures result
        \\    {
        \\        let slot: StorageSlot = dataSlot(owner);
        \\        let data: [u256; 2] = [first, second];
        \\        fixed.store(slot, data);
        \\        let loaded: [u256; 2] = fixed.load(slot, 2);
        \\        return loaded[0] == first && loaded[1] == second && fixed.hasData(slot, 2);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "roundtrip");
    defer result.deinit(testing.allocator);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.error_kinds);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(result.success);
}

test "verification frames computed storage unrelated word offsets" {
    const source_text =
        \\contract C {
        \\    pub fn offset_frame(owner: address, value: u256) -> bool
        \\        modifies @storageRange(@storageDerive("verify.computed.frame.offset", owner), 2)
        \\        ensures result
        \\    {
        \\        let slot: StorageSlot = @storageDerive("verify.computed.frame.offset", owner);
        \\        let before: u256 = @storageWordLoad(slot, 1);
        \\        @storageWordStore(slot, 0, value);
        \\        let after: u256 = @storageWordLoad(slot, 1);
        \\        return after == before;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "offset_frame");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames computed storage unrelated namespaces" {
    const source_text =
        \\contract C {
        \\    pub fn namespace_frame(owner: address, value: u256) -> bool
        \\        modifies @storageRange(@storageDerive("verify.computed.frame.a", owner), 1)
        \\        ensures result
        \\    {
        \\        let a: StorageSlot = @storageDerive("verify.computed.frame.a", owner);
        \\        let b: StorageSlot = @storageDerive("verify.computed.frame.b", owner);
        \\        let before: u256 = @storageWordLoad(b, 0);
        \\        @storageWordStore(a, 0, value);
        \\        let after: u256 = @storageWordLoad(b, 0);
        \\        return after == before;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "namespace_frame");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification models bounded computed storage range erase" {
    const source_text =
        \\contract C {
        \\    pub fn erase_then_read(owner: address, first: u256, second: u256) -> bool
        \\        modifies @storageRange(@storageDerive("verify.computed.erase", owner), 2)
        \\        ensures result
        \\    {
        \\        let slot: StorageSlot = @storageDerive("verify.computed.erase", owner);
        \\        @storageWordStore(slot, 0, first);
        \\        @storageWordStore(slot, 1, second);
        \\        @storageRangeErase(@storageRange(slot, 2));
        \\        return @storageWordLoad(slot, 0) == 0 && @storageWordLoad(slot, 1) == 0;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "erase_then_read");
    defer result.deinit(testing.allocator);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expectEqualStrings("", result.error_kinds);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expect(result.success);
}

test "verification frames computed storage offsets outside erased range" {
    const source_text =
        \\contract C {
        \\    pub fn erase_frame(owner: address) -> bool
        \\        modifies @storageRange(@storageDerive("verify.computed.erase.frame", owner), 2)
        \\        ensures result
        \\    {
        \\        let slot: StorageSlot = @storageDerive("verify.computed.erase.frame", owner);
        \\        let before: u256 = @storageWordLoad(slot, 2);
        \\        @storageRangeErase(@storageRange(slot, 2));
        \\        let after: u256 = @storageWordLoad(slot, 2);
        \\        return after == before;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "erase_frame");
    defer result.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expect(result.success);
}

test "verification does not assume runtime keccak256 collision resistance" {
    const source_text =
        \\pub fn hash_collision_claim(a: bytes, b: bytes)
        \\    ensures(@keccak256(a) == @keccak256(b))
        \\{
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "hash_collision_claim");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification supports native string and bytes len field access under bounded byte-sequence model" {
    const source_text =
        \\pub fn same_string_len(text: string) -> bool
        \\    ensures(result)
        \\{
        \\    return text.len == text.len;
        \\}
        \\
        \\pub fn same_bytes_len(data: bytes) -> bool
        \\    ensures(result)
        \\{
        \\    return data.len == data.len;
        \\}
    ;

    var string_result = try verifyTextWithoutDegradation(source_text, "same_string_len");
    defer string_result.deinit(testing.allocator);
    try testing.expect(string_result.success);
    try testing.expect(!string_result.degraded);

    var bytes_result = try verifyTextWithoutDegradation(source_text, "same_bytes_len");
    defer bytes_result.deinit(testing.allocator);
    try testing.expect(bytes_result.success);
    try testing.expect(!bytes_result.degraded);
}

test "verification supports native string and bytes index access when length preconditions establish safety" {
    const source_text =
        \\pub fn first_eq(data: bytes, text: string) -> bool
        \\    requires(data.len > 0)
        \\    requires(text.len > 0)
        \\    ensures(result == (data[0] == text[0]))
        \\{
        \\    return data[0] == text[0];
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "first_eq");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
}

test "verification requires byte slice bounds before using projected length" {
    const source_text =
        \\pub fn cut_unchecked(data: bytes) -> bytes
        \\{
        \\    return @slice(data, 0, 1);
        \\}
        \\
        \\pub fn cut_checked(data: bytes) -> bytes
        \\    requires(data.len >= 4)
        \\    ensures(result.len == 3)
        \\{
        \\    return @slice(data, 1, 3);
        \\}
    ;

    var unchecked = try verifyTextWithoutDegradationWithTimeout(source_text, "cut_unchecked", 5_000);
    defer unchecked.deinit(testing.allocator);
    try testing.expect(unchecked.errors_len > 0);
    try testing.expectEqualStrings("InvariantViolation", unchecked.error_kinds);
    try testing.expect(!unchecked.degraded);

    var checked = try verifyTextWithoutDegradationWithTimeout(source_text, "cut_checked", 5_000);
    defer checked.deinit(testing.allocator);
    try testing.expect(checked.success);
    try testing.expect(!checked.degraded);
}

test "verification requires byte concat bounds before using projected length" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\pub fn join_unchecked(a: bytes, b: bytes) -> bytes
        \\{
        \\    return a + b;
        \\}
        \\
        \\pub fn join_checked(a: bytes, b: bytes) -> bytes
        \\    requires(b.len <= std.constants.U256_MAX - 32)
        \\    requires(a.len <= std.constants.U256_MAX - 32 - b.len)
        \\    ensures(result.len == a.len + b.len)
        \\{
        \\    return @concat(a, b);
        \\}
    ;

    var unchecked = try verifyTextWithoutDegradationWithTimeout(source_text, "join_unchecked", 5_000);
    defer unchecked.deinit(testing.allocator);
    try testing.expect(unchecked.errors_len > 0);
    try testing.expectEqualStrings("InvariantViolation", unchecked.error_kinds);
    try testing.expect(!unchecked.degraded);

    var checked = try verifyTextWithoutDegradationWithTimeout(source_text, "join_checked", 5_000);
    defer checked.deinit(testing.allocator);
    try testing.expect(checked.success);
    try testing.expect(!checked.degraded);
}

test "verification checks safety obligations from branch conditions" {
    const source_text =
        \\pub fn branch_div_unchecked(x: u256) -> u256
        \\{
        \\    if (1 / x == 0) {
        \\        return 1;
        \\    }
        \\    return 2;
        \\}
        \\
        \\pub fn branch_div_checked(x: u256) -> u256
        \\    requires(x != 0)
        \\{
        \\    if (1 / x == 0) {
        \\        return 1;
        \\    }
        \\    return 2;
        \\}
    ;

    var unchecked = try verifyTextWithoutDegradationWithTimeout(source_text, "branch_div_unchecked", 5_000);
    defer unchecked.deinit(testing.allocator);
    try testing.expect(!unchecked.success);
    try testing.expect(unchecked.errors_len > 0);
    try testing.expectEqualStrings("InvariantViolation", unchecked.error_kinds);
    try testing.expect(!unchecked.degraded);

    var checked = try verifyTextWithoutDegradationWithTimeout(source_text, "branch_div_checked", 5_000);
    defer checked.deinit(testing.allocator);
    try testing.expect(checked.success);
    try testing.expect(!checked.degraded);
}

test "verification proves checked power for a bounded safe case without degradation" {
    const source_text =
        \\pub fn square_ten(x: u8) -> u8
        \\    requires(x <= 10)
        \\    ensures(result == x * x)
        \\{
        \\    return x ** 2;
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithTimeout(source_text, "square_ten", 5_000);
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
}

test "verification rejects unchecked power overflow without degradation" {
    const source_text =
        \\pub fn square_any(x: u8) -> u8
        \\{
        \\    return x ** 2;
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithTimeout(source_text, "square_any", 5_000);
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("InvariantViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification proves checked power for bounded symbolic exponent without degradation" {
    const source_text =
        \\pub fn bounded_pow(base: u8, exp: u8) -> u8
        \\    requires(base <= 3)
        \\    requires(exp <= 4)
        \\    ensures(result <= 81)
        \\{
        \\    return base ** exp;
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithTimeout(source_text, "bounded_pow", 5_000);
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
}

test "verification rejects symbolic checked power overflow without degradation" {
    const source_text =
        \\pub fn bounded_pow_overflow(base: u8, exp: u8) -> u8
        \\    requires(exp <= 4)
        \\{
        \\    return base ** exp;
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithTimeout(source_text, "bounded_pow_overflow", 5_000);
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("InvariantViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification ignores folded-only private helpers but keeps runtime comptime specializations" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Sample {
        \\    fn add(a: u256, b: u256) -> u256 { return a + b; }
        \\    fn mul(a: u256, b: u256) -> u256 { return a * b; }
        \\
        \\    fn sum_of_squares(a: u256, b: u256) -> u256 {
        \\        return add(mul(a, a), mul(b, b));
        \\    }
        \\
        \\    pub fn folded_only() -> u256 {
        \\        const value: u256 = sum_of_squares(3, 4);
        \\        return value;
        \\    }
        \\
        \\    fn align_up(comptime alignment: u256, value: u256) -> u256
        \\        requires(alignment > 0)
        \\        requires(value <= std.constants.U256_MAX - (alignment - 1))
        \\    {
        \\        const mask: u256 = alignment - 1;
        \\        return (value + mask) - ((value + mask) % alignment);
        \\    }
        \\
        \\    pub fn runtime_specialized(x: u256) -> u256 {
        \\        return align_up(32, x);
        \\    }
        \\
        \\    pub fn runtime_specialized_checked(x: u256) -> u256
        \\        requires(x <= std.constants.U256_MAX - 31)
        \\    {
        \\        return align_up(32, x);
        \\    }
        \\}
    ;

    var folded = try verifyTextWithoutDegradationWithTimeout(source_text, "folded_only", 5_000);
    defer folded.deinit(testing.allocator);
    try testing.expect(folded.success);
    try testing.expect(!folded.degraded);

    var specialized = try verifyTextWithoutDegradationWithTimeout(source_text, "runtime_specialized", 5_000);
    defer specialized.deinit(testing.allocator);
    try testing.expect(!specialized.success);
    try testing.expect(std.mem.indexOf(u8, specialized.error_kinds, "PreconditionViolation") != null);
    try testing.expect(!specialized.degraded);

    var checked = try verifyTextWithoutDegradationWithTimeout(source_text, "runtime_specialized_checked", 5_000);
    defer checked.deinit(testing.allocator);
    try testing.expect(checked.success);
    try testing.expect(!checked.degraded);
}

test "verification accepts requires helper calls that read storage without degradation" {
    const source_text =
        \\contract Sample {
        \\    storage var total_deposits: u256 = 0;
        \\
        \\    fn dep() -> u256 {
        \\        return total_deposits;
        \\    }
        \\
        \\    pub fn borrow(amount: u256) -> bool
        \\        requires(amount > 0)
        \\        requires(amount <= dep())
        \\    {
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "borrow");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expect(!result.degraded);
}

test "verification frames storage outside internal callee modifies set" {
    const source_text =
        \\contract V {
        \\    storage var stable: u256 = 7;
        \\    storage var touched: u256 = 0;
        \\
        \\    fn bump(next: u256)
        \\        modifies touched
        \\    {
        \\        touched = next;
        \\    }
        \\
        \\    pub fn f(next: u256) -> bool
        \\        ensures(stable == old(stable))
        \\    {
        \\        bump(next);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames distinct map keys across internal callee modifies set" {
    const source_text =
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    fn set_balance(user: address, value: u256)
        \\        modifies balances[user]
        \\    {
        \\        balances[user] = value;
        \\    }
        \\
        \\    pub fn f(user: address, other: address, value: u256) -> bool
        \\        requires(user != other)
        \\        ensures(balances[other] == old(balances[other]))
        \\    {
        \\        set_balance(user, value);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification does not frame possibly aliased map keys across internal callee modifies set" {
    const source_text =
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    fn set_balance(user: address, value: u256)
        \\        modifies balances[user]
        \\    {
        \\        balances[user] = value;
        \\    }
        \\
        \\    pub fn f(user: address, other: address, value: u256) -> bool
        \\        ensures(balances[other] == old(balances[other]))
        \\    {
        \\        set_balance(user, value);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification uses opaque modifies metadata when internal summary inlining is disabled" {
    const source_text =
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    fn set_balance(user: address, value: u256)
        \\        modifies balances[user]
        \\    {
        \\        balances[user] = value;
        \\    }
        \\
        \\    pub fn f(user: address, other: address, value: u256) -> bool
        \\        requires(user != other)
        \\        ensures(balances[other] == old(balances[other]))
        \\    {
        \\        set_balance(user, value);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithSummaryInlineDepth(source_text, "f", 0);
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expectEqualStrings("", result.precision_notes);
}

test "verification opaque modifies metadata does not frame declared written key" {
    const source_text =
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    fn set_balance(user: address, value: u256)
        \\        modifies balances[user]
        \\    {
        \\        balances[user] = value;
        \\    }
        \\
        \\    pub fn f(user: address, value: u256) -> bool
        \\        ensures(balances[user] == old(balances[user]))
        \\    {
        \\        set_balance(user, value);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithSummaryInlineDepth(source_text, "f", 0);
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", result.error_kinds);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expectEqualStrings("", result.precision_notes);
}

test "compiler marks imported-module calls as summary-boundary candidates" {
    var compilation = try compilePackage("ora-example/smt/modifies/pass_imported_summary_map_key_frame.ora");
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.imported_call"));
}

test "verification uses imported-module opaque modifies metadata by default" {
    var result = try verifyPackageWithoutDegradation(
        "ora-example/smt/modifies/pass_imported_summary_map_key_frame.ora",
        "f",
    );
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expectEqualStrings("", result.precision_notes);
}

test "verification imported-summary mode is discriminated from exact imported body inlining" {
    var exact = try verifyPackageWithoutDegradationWithImportedSummaryMode(
        "ora-example/smt/modifies/fail_imported_summary_discriminator.ora",
        "f",
        false,
    );
    defer exact.deinit(testing.allocator);
    try testing.expect(!exact.success);
    try testing.expectEqualStrings("InvariantViolation", exact.error_kinds);
    try testing.expect(!exact.degraded);

    var summary_only = try verifyPackageWithoutDegradation(
        "ora-example/smt/modifies/fail_imported_summary_discriminator.ora",
        "f",
    );
    defer summary_only.deinit(testing.allocator);
    try testing.expect(summary_only.success);
    try testing.expect(!summary_only.degraded);
    try testing.expectEqual(@as(usize, 0), summary_only.errors_len);
    try testing.expectEqualStrings("", summary_only.soundness_losses);
    try testing.expectEqualStrings("", summary_only.precision_notes);
}

test "verification frames distinct nested map keys across internal callee modifies set" {
    const source_text =
        \\contract V {
        \\    storage var allowances: map<address, map<address, u256>>;
        \\
        \\    fn set_allowance(owner: address, spender: address, value: u256)
        \\        modifies allowances[owner][spender]
        \\    {
        \\        allowances[owner][spender] = value;
        \\    }
        \\
        \\    pub fn f(
        \\        owner: address,
        \\        spender: address,
        \\        other_owner: address,
        \\        other_spender: address,
        \\        value: u256,
        \\    ) -> bool
        \\        requires(owner != other_owner)
        \\        ensures(allowances[other_owner][other_spender] == old(allowances[other_owner][other_spender]))
        \\    {
        \\        set_allowance(owner, spender, value);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames distinct struct fields across internal callee modifies set" {
    const source_text =
        \\contract V {
        \\    struct Config {
        \\        owner: address,
        \\        admin: address,
        \\    }
        \\
        \\    storage var config: Config;
        \\
        \\    fn set_owner(next_owner: address)
        \\        modifies config.owner
        \\    {
        \\        config.owner = next_owner;
        \\    }
        \\
        \\    pub fn f(next_owner: address) -> bool
        \\        ensures(config.admin == old(config.admin))
        \\    {
        \\        set_owner(next_owner);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification preserves callee requires when summary inlining falls back" {
    const source_text =
        \\contract V {
        \\    fn require_nonzero(x: u256)
        \\        requires(x != 0)
        \\    {
        \\    }
        \\
        \\    pub fn unchecked(x: u256) -> u256 {
        \\        require_nonzero(x);
        \\        return 1;
        \\    }
        \\
        \\    pub fn checked(x: u256) -> u256
        \\        requires(x != 0)
        \\    {
        \\        require_nonzero(x);
        \\        return 1;
        \\    }
        \\}
    ;

    var unchecked = try verifyTextWithoutDegradationWithSummaryInlineDepth(source_text, "unchecked", 0);
    defer unchecked.deinit(testing.allocator);
    try testing.expect(!unchecked.success);
    try testing.expect(!unchecked.degraded);
    try testing.expect(unchecked.errors_len > 0);
    try testing.expect(std.mem.indexOf(u8, unchecked.error_kinds, "PreconditionViolation") != null);

    var checked = try verifyTextWithoutDegradationWithSummaryInlineDepth(source_text, "checked", 0);
    defer checked.deinit(testing.allocator);
    try testing.expect(checked.success);
    try testing.expect(!checked.degraded);
    try testing.expectEqual(@as(usize, 0), checked.errors_len);
}

test "verification propagates inline requires and ensures through callers" {
    const source_text =
        \\contract V {
        \\    inline fn plusOne(value: u256) -> u256
        \\        requires(value < 10)
        \\        ensures result == value + 1
        \\    {
        \\        return value + 1;
        \\    }
        \\
        \\    pub fn unchecked(value: u256) -> u256 {
        \\        return plusOne(value);
        \\    }
        \\
        \\    pub fn checked(value: u256) -> u256
        \\        requires(value < 10)
        \\        ensures result > value
        \\    {
        \\        return plusOne(value);
        \\    }
        \\}
    ;

    var unchecked = try verifyTextWithoutDegradation(source_text, "unchecked");
    defer unchecked.deinit(testing.allocator);
    try testing.expect(!unchecked.success);
    try testing.expect(!unchecked.degraded);
    try testing.expect(unchecked.errors_len > 0);
    try testing.expectEqualStrings("PreconditionViolation", unchecked.error_kinds);
    try testing.expectEqualStrings("", unchecked.soundness_losses);

    var checked = try verifyTextWithoutDegradation(source_text, "checked");
    defer checked.deinit(testing.allocator);
    try testing.expect(checked.success);
    try testing.expect(!checked.degraded);
    try testing.expectEqual(@as(usize, 0), checked.errors_len);
    try testing.expectEqualStrings("", checked.soundness_losses);
}

test "verification propagates fallible inline requires and ensures_ok through callers" {
    const source_text =
        \\error Failure;
        \\
        \\contract V {
        \\    inline fn plusOne(value: u256) -> !u256 | Failure
        \\        requires(value < 10)
        \\        ensures_ok(result == value + 1)
        \\    {
        \\        return value + 1;
        \\    }
        \\
        \\    pub fn unchecked(value: u256) -> !u256 | Failure {
        \\        let out: u256 = try plusOne(value);
        \\        return out;
        \\    }
        \\
        \\    pub fn checked(value: u256) -> !u256 | Failure
        \\        requires(value < 10)
        \\        ensures_ok(result > value)
        \\    {
        \\        let out: u256 = try plusOne(value);
        \\        return out;
        \\    }
        \\}
    ;

    var unchecked = try verifyTextWithoutDegradation(source_text, "unchecked");
    defer unchecked.deinit(testing.allocator);
    try testing.expect(!unchecked.success);
    try testing.expect(!unchecked.degraded);
    try testing.expect(unchecked.errors_len > 0);
    try testing.expectEqualStrings("PreconditionViolation", unchecked.error_kinds);
    try testing.expectEqualStrings("", unchecked.soundness_losses);

    var checked = try verifyTextWithoutDegradation(source_text, "checked");
    defer checked.deinit(testing.allocator);
    try testing.expect(checked.success);
    try testing.expect(!checked.degraded);
    try testing.expectEqual(@as(usize, 0), checked.errors_len);
    try testing.expectEqualStrings("", checked.soundness_losses);
}

test "verification uses opaque modifies metadata for struct fields when internal summary inlining is disabled" {
    const source_text =
        \\contract V {
        \\    struct Config {
        \\        owner: address,
        \\        admin: address,
        \\    }
        \\
        \\    storage var config: Config;
        \\
        \\    fn set_owner(next_owner: address)
        \\        modifies config.owner
        \\    {
        \\        config.owner = next_owner;
        \\    }
        \\
        \\    pub fn f(next_owner: address) -> bool
        \\        ensures(config.admin == old(config.admin))
        \\    {
        \\        set_owner(next_owner);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithSummaryInlineDepth(source_text, "f", 0);
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expectEqualStrings("", result.precision_notes);
}

test "verification opaque modifies metadata does not frame declared written struct field" {
    const source_text =
        \\contract V {
        \\    struct Config {
        \\        owner: address,
        \\        admin: address,
        \\    }
        \\
        \\    storage var config: Config;
        \\
        \\    fn set_owner(next_owner: address)
        \\        modifies config.owner
        \\    {
        \\        config.owner = next_owner;
        \\    }
        \\
        \\    pub fn f(next_owner: address) -> bool
        \\        ensures(config.owner == old(config.owner))
        \\    {
        \\        set_owner(next_owner);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithSummaryInlineDepth(source_text, "f", 0);
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("PostconditionViolation", result.error_kinds);
    try testing.expectEqualStrings("", result.soundness_losses);
    try testing.expectEqualStrings("", result.precision_notes);
}

test "verification does not let internal modifies hide unresolved external calls" {
    const source_text =
        \\extern trait External {
        \\    call fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var stable: u256 = 7;
        \\
        \\    fn helper(target: address) -> !bool | ExternalCallFailed
        \\        modifies()
        \\    {
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        ensures(stable == old(stable))
        \\    {
        \\        return helper(target);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.degraded);
    try testing.expect(std.mem.containsAtLeast(u8, result.soundness_losses, 1, "unresolved_callee"));
}

test "verification rejects unresolved external call preserving storage" {
    const source_text =
        \\extern trait External {
        \\    call fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var s: u256 = 0;
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        ensures(s == old(s))
        \\    {
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expect(std.mem.containsAtLeast(u8, result.soundness_losses, 1, "unresolved_callee"));
}

test "verification frames storage across unresolved staticcall" {
    const source_text =
        \\extern trait External {
        \\    staticcall fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var s: u256 = 0;
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        ensures(s == old(s))
        \\    {
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames storage across staticcall inside modifies empty function" {
    const source_text =
        \\extern trait External {
        \\    staticcall fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var stable: u256 = 0;
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        modifies()
        \\        ensures(stable == old(stable))
        \\    {
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames locked storage across unresolved external call" {
    const source_text =
        \\extern trait External {
        \\    call fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var stable: u256 = 7;
        \\    storage var other: u256 = 11;
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        ensures(stable == old(stable))
        \\    {
        \\        @lock(stable);
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification does not frame unlocked storage across locked external call" {
    const source_text =
        \\extern trait External {
        \\    call fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var stable: u256 = 7;
        \\    storage var other: u256 = 11;
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        ensures(other == old(other))
        \\    {
        \\        @lock(stable);
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames locked map key across unresolved external call" {
    const source_text =
        \\extern trait External {
        \\    call fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    pub fn f(target: address, user: address) -> !bool | ExternalCallFailed
        \\        ensures(balances[user] == old(balances[user]))
        \\    {
        \\        @lock(balances[user]);
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification does not frame unlocked map key across locked external call" {
    const source_text =
        \\extern trait External {
        \\    call fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    pub fn f(target: address, user: address, other: address) -> !bool | ExternalCallFailed
        \\        requires(user != other)
        \\        ensures(balances[other] == old(balances[other]))
        \\    {
        \\        @lock(balances[user]);
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification corpus covers smt modifies framing matrix" {
    const Probe = struct {
        path: []const u8,
        expect_success: bool,
        expect_degraded: bool = false,
        expected_loss: []const u8 = "",
        expected_error_kinds: []const u8 = "",
    };

    const probes = [_]Probe{
        .{
            .path = "ora-example/smt/modifies/pass_internal_root_frame.ora",
            .expect_success = true,
        },
        .{
            .path = "ora-example/smt/modifies/pass_internal_map_key_frame.ora",
            .expect_success = true,
        },
        .{
            .path = "ora-example/smt/modifies/fail_internal_map_key_alias.ora",
            .expect_success = false,
            .expected_error_kinds = "PostconditionViolation",
        },
        .{
            .path = "ora-example/smt/modifies/pass_internal_nested_map_frame.ora",
            .expect_success = true,
        },
        .{
            .path = "ora-example/smt/modifies/pass_internal_struct_field_frame.ora",
            .expect_success = true,
        },
        .{
            .path = "ora-example/smt/modifies/pass_staticcall_modifies_empty_frame.ora",
            .expect_success = true,
        },
        .{
            .path = "ora-example/smt/modifies/fail_modifies_empty_unresolved_call.ora",
            .expect_success = false,
            .expect_degraded = true,
            .expected_loss = "unresolved_callee",
            .expected_error_kinds = "EncodingDegraded",
        },
        .{
            .path = "ora-example/smt/modifies/pass_locked_call_root_frame.ora",
            .expect_success = true,
        },
        .{
            .path = "ora-example/smt/modifies/fail_locked_call_unlocked_root.ora",
            .expect_success = false,
            .expected_error_kinds = "PostconditionViolation",
        },
        .{
            .path = "ora-example/smt/modifies/pass_locked_call_map_key_frame.ora",
            .expect_success = true,
        },
        .{
            .path = "ora-example/smt/modifies/fail_locked_call_unlocked_map_key.ora",
            .expect_success = false,
            .expected_error_kinds = "PostconditionViolation",
        },
    };

    for (probes) |probe| {
        var seq_result = try verifyExampleWithoutDegradation(probe.path, "f", false, 5_000);
        defer seq_result.deinit(testing.allocator);

        var par_result = try verifyExampleWithoutDegradation(probe.path, "f", true, 5_000);
        defer par_result.deinit(testing.allocator);

        try testing.expectEqual(probe.expect_success, seq_result.success);
        try testing.expectEqual(probe.expect_success, par_result.success);
        try testing.expectEqual(probe.expect_degraded, seq_result.degraded);
        try testing.expectEqual(probe.expect_degraded, par_result.degraded);
        if (probe.expected_loss.len != 0) {
            try testing.expect(std.mem.containsAtLeast(u8, seq_result.soundness_losses, 1, probe.expected_loss));
            try testing.expect(std.mem.containsAtLeast(u8, par_result.soundness_losses, 1, probe.expected_loss));
        } else {
            try testing.expectEqualStrings("", seq_result.soundness_losses);
            try testing.expectEqualStrings("", par_result.soundness_losses);
        }
        if (probe.expect_success) {
            try testing.expectEqual(@as(usize, 0), seq_result.errors_len);
            try testing.expectEqual(@as(usize, 0), par_result.errors_len);
        } else {
            try testing.expect(seq_result.errors_len > 0);
            try testing.expect(par_result.errors_len > 0);
            try testing.expectEqualStrings(probe.expected_error_kinds, seq_result.error_kinds);
            try testing.expectEqualStrings(probe.expected_error_kinds, par_result.error_kinds);
        }
        try testing.expectEqualStrings("", seq_result.precision_notes);
        try testing.expectEqualStrings("", par_result.precision_notes);
        try expectVerificationProbeEquivalent(&seq_result, &par_result);
    }
}

test "verification does not assume unresolved staticcall return value" {
    const source_text =
        \\extern trait External {
        \\    staticcall fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    pub fn f(target: address) -> bool
        \\        ensures(result)
        \\    {
        \\        try {
        \\            let r: bool = try external<External>(target, gas: 50000).x();
        \\            return r;
        \\        } catch {
        \\            return true;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification treats unresolved staticcall return as deterministic" {
    const source_text =
        \\extern trait External {
        \\    staticcall fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    pub fn f(target: address) -> bool
        \\        ensures(result == result)
        \\    {
        \\        try {
        \\            let r: bool = try external<External>(target, gas: 50000).x();
        \\            return r;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification uses trusted extern trait scalar ensures summary" {
    const source_text =
        \\extern trait Oracle {
        \\    staticcall fn quote(self) -> u256
        \\        ensures(result == 42);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> !bool | ExternalCallFailed
        \\        ensures(observed == 42)
        \\    {
        \\        let q: u256 = try external<Oracle>(target, gas: 50000).quote();
        \\        observed = q;
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification uses trusted extern trait struct ensures summary" {
    const source_text =
        \\struct Snapshot {
        \\    current: u256,
        \\    tag: u256,
        \\}
        \\
        \\extern trait Target {
        \\    staticcall fn snapshot(self) -> Snapshot
        \\        ensures(result.current == 42)
        \\        ensures(result.tag == 7);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed_current: u256 = 0;
        \\    storage var observed_tag: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> !bool | ExternalCallFailed
        \\        ensures(observed_current == 42)
        \\        ensures(observed_tag == 7)
        \\    {
        \\        let snapshot: Snapshot = try external<Target>(target, gas: 50000).snapshot();
        \\        observed_current = snapshot.current;
        \\        observed_tag = snapshot.tag;
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification carries trusted extern trait scalar summary through try catch" {
    const source_text =
        \\extern trait Oracle {
        \\    staticcall fn quote(self) -> u256
        \\        ensures(result == 42);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> bool
        \\        ensures(!result or observed == 42)
        \\    {
        \\        try {
        \\            let q: u256 = try external<Oracle>(target, gas: 50000).quote();
        \\            observed = q;
        \\            return true;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification proves trusted extern trait requires through try catch" {
    const source_text =
        \\extern trait Oracle {
        \\    staticcall fn quote(self, amount: u256) -> u256
        \\        requires(amount == 42)
        \\        ensures(result == amount);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed: u256 = 0;
        \\
        \\    pub fn pull(target: address, amount: u256) -> bool
        \\        requires(amount == 42)
        \\        ensures(!result or observed == 42)
        \\    {
        \\        try {
        \\            let q: u256 = try external<Oracle>(target, gas: 50000).quote(amount);
        \\            observed = q;
        \\            return true;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rejects unproven trusted extern trait requires through try catch" {
    const source_text =
        \\extern trait Oracle {
        \\    staticcall fn quote(self, amount: u256) -> u256
        \\        requires(amount == 42)
        \\        ensures(result == amount);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed: u256 = 0;
        \\
        \\    pub fn pull(target: address, amount: u256) -> bool
        \\        ensures(!result or observed == amount)
        \\    {
        \\        try {
        \\            let q: u256 = try external<Oracle>(target, gas: 50000).quote(amount);
        \\            observed = q;
        \\            return true;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
}

test "verification carries trusted extern trait struct summary through try catch" {
    const source_text =
        \\struct Snapshot {
        \\    current: u256,
        \\    tag: u256,
        \\}
        \\
        \\extern trait Target {
        \\    staticcall fn snapshot(self) -> Snapshot
        \\        ensures(result.current == 42)
        \\        ensures(result.tag == 7);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed_current: u256 = 0;
        \\    storage var observed_tag: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> bool
        \\        ensures(!result or observed_current == 42)
        \\        ensures(!result or observed_tag == 7)
        \\    {
        \\        try {
        \\            let snapshot: Snapshot = try external<Target>(target, gas: 50000).snapshot();
        \\            observed_current = snapshot.current;
        \\            observed_tag = snapshot.tag;
        \\            return true;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "extern trait requires cannot reference result" {
    const source_text =
        \\extern trait Oracle {
        \\    staticcall fn quote(self) -> u256
        \\        requires(result == 42);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(diagnosticMessagesContain(type_diags, "`result` is only available in ensures clauses"));
}

test "verification carries trusted extern trait tuple summary through try catch" {
    const source_text =
        \\extern trait Target {
        \\    staticcall fn quote(self) -> (u256, bool)
        \\        ensures(result.0 == 42)
        \\        ensures(result.1);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> bool
        \\        ensures(!result or observed == 42)
        \\    {
        \\        try {
        \\            let quote: (u256, bool) = try external<Target>(target, gas: 50000).quote();
        \\            observed = quote.0;
        \\            return quote.1;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification carries trusted extern trait bytes length summary through try catch" {
    const source_text =
        \\extern trait Target {
        \\    staticcall fn payload(self) -> bytes
        \\        ensures(result.len == 4);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed_len: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> bool
        \\        ensures(!result or observed_len == 4)
        \\    {
        \\        try {
        \\            let payload: bytes = try external<Target>(target, gas: 50000).payload();
        \\            observed_len = payload.len;
        \\            return true;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradationWithTimeout(source_text, "pull", 5_000);
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification carries trusted extern trait three-field struct summary through try catch" {
    const source_text =
        \\struct Snapshot {
        \\    current: u256,
        \\    tag: u256,
        \\    nonce: u256,
        \\}
        \\
        \\extern trait Target {
        \\    staticcall fn snapshot(self) -> Snapshot
        \\        ensures(result.current == 42)
        \\        ensures(result.tag == 7)
        \\        ensures(result.nonce == 99);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed_current: u256 = 0;
        \\    storage var observed_tag: u256 = 0;
        \\    storage var observed_nonce: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> bool
        \\        ensures(!result or observed_current == 42)
        \\        ensures(!result or observed_tag == 7)
        \\        ensures(!result or observed_nonce == 99)
        \\    {
        \\        try {
        \\            let snapshot: Snapshot = try external<Target>(target, gas: 50000).snapshot();
        \\            observed_current = snapshot.current;
        \\            observed_tag = snapshot.tag;
        \\            observed_nonce = snapshot.nonce;
        \\            return true;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification carries trusted extern trait nested tuple struct summary through try catch" {
    const source_text =
        \\struct Snapshot {
        \\    current: u256,
        \\    ok: bool,
        \\}
        \\
        \\extern trait Target {
        \\    staticcall fn quote(self, amount: u256) -> (Snapshot, u256)
        \\        ensures(result.0.current == amount)
        \\        ensures(result.0.ok)
        \\        ensures(result.1 == amount);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var observed: u256 = 0;
        \\    storage var mirror: u256 = 0;
        \\
        \\    pub fn pull(target: address, amount: u256) -> bool
        \\        ensures(!result or observed == amount)
        \\        ensures(!result or mirror == amount)
        \\    {
        \\        try {
        \\            let quote: (Snapshot, u256) = try external<Target>(target, gas: 50000).quote(amount);
        \\            observed = quote.0.current;
        \\            mirror = quote.1;
        \\            return quote.0.ok;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames caller storage through trusted extern call summary" {
    const source_text =
        \\extern trait External {
        \\    call fn quote(self) -> u256
        \\        ensures(result == 42);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var stable: u256 = 7;
        \\    storage var observed: u256 = 0;
        \\
        \\    pub fn pull(target: address) -> bool
        \\        ensures(!result or stable == old(stable))
        \\        ensures(!result or observed == 42)
        \\    {
        \\        try {
        \\            let q: u256 = try external<External>(target, gas: 50000).quote();
        \\            observed = q;
        \\            return true;
        \\        } catch {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames caller storage through trusted extern call summary in helper" {
    const source_text =
        \\extern trait External {
        \\    call fn quote(self) -> u256
        \\        ensures(result == 42);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var stable: u256 = 7;
        \\
        \\    fn observe(target: address) -> !u256 | ExternalCallFailed {
        \\        return external<External>(target, gas: 50000).quote();
        \\    }
        \\
        \\    pub fn pull(target: address) -> !bool | ExternalCallFailed
        \\        ensures(stable == old(stable))
        \\    {
        \\        let ignored: u256 = try observe(target);
        \\        return true;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "pull");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification frames storage through known helper staticcall" {
    const source_text =
        \\extern trait External {
        \\    staticcall fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var s: u256 = 0;
        \\
        \\    fn observe(target: address) -> !bool | ExternalCallFailed {
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        ensures(s == old(s))
        \\    {
        \\        return observe(target);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rejects storage preservation through known helper external call" {
    const source_text =
        \\extern trait External {
        \\    call fn x(self) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract V {
        \\    storage var s: u256 = 0;
        \\
        \\    fn observe(target: address) -> !bool | ExternalCallFailed {
        \\        return external<External>(target, gas: 50000).x();
        \\    }
        \\
        \\    pub fn f(target: address) -> !bool | ExternalCallFailed
        \\        ensures(s == old(s))
        \\    {
        \\        return observe(target);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expect(std.mem.containsAtLeast(u8, result.soundness_losses, 1, "unresolved_callee"));
}

test "verification rebinds pure callee old storage to call-site state" {
    const source_text =
        \\contract V {
        \\    storage var s: u256 = 0;
        \\
        \\    fn observe() -> u256
        \\        ensures(result == old(s))
        \\    {
        \\        return s;
        \\    }
        \\
        \\    pub fn f(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        s = value;
        \\        return observe();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rebinds pure callee old mapping storage to call-site state" {
    const source_text =
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    fn observe(account: address) -> u256
        \\        ensures(result == old(balances[account]))
        \\    {
        \\        return balances[account];
        \\    }
        \\
        \\    pub fn f(account: address, value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        balances[account] = value;
        \\        return observe(account);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rebinds pure callee old storage for multiple slots" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var left: u256 = 0;
        \\    storage var right: u256 = 0;
        \\
        \\    fn observe() -> u256
        \\        requires(old(left) <= std.constants.U256_MAX - old(right))
        \\        ensures(result == old(left) + old(right))
        \\    {
        \\        return left + right;
        \\    }
        \\
        \\    pub fn f(a: u256, b: u256) -> u256
        \\        requires(a <= std.constants.U256_MAX - b)
        \\        ensures(result == a + b)
        \\    {
        \\        left = a;
        \\        right = b;
        \\        return observe();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rebinds nested pure callee old storage to call-site state" {
    const source_text =
        \\contract V {
        \\    storage var s: u256 = 0;
        \\
        \\    fn leaf() -> u256
        \\        ensures(result == old(s))
        \\    {
        \\        return s;
        \\    }
        \\
        \\    fn observe() -> u256
        \\        ensures(result == old(s))
        \\    {
        \\        return leaf();
        \\    }
        \\
        \\    pub fn f(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        s = value;
        \\        return observe();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rebinds stateful callee old storage to call-site state" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var s: u256 = 0;
        \\
        \\    fn bump()
        \\        requires(s < std.constants.U256_MAX)
        \\        ensures(s == old(s) + 1)
        \\    {
        \\        s = s + 1;
        \\    }
        \\
        \\    pub fn f(value: u256)
        \\        requires(value < std.constants.U256_MAX)
        \\        ensures(s == value + 1)
        \\    {
        \\        s = value;
        \\        bump();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification treats old in loop invariants as function entry state" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total <= std.constants.U256_MAX - 101)
        \\        ensures(total == old(total) + 1)
        \\    {
        \\        total = total + 1;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total == old(total) + 1
        \\        {
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification supports explicit loop-entry snapshot idiom" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total <= std.constants.U256_MAX - 101)
        \\        ensures(total == old(total) + 1)
        \\    {
        \\        total = total + 1;
        \\        let loop_start: u256 = total;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total == loop_start
        \\        {
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

fn expectExplainModeNoInconsistentAssumptionNote(source_text: []const u8) !void {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.filter_function_name = "f";
    verifier.setExplainCores(true);

    var result = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);

    var artifacts = try verifier.buildSmtReport(hir_result.module.raw_module, "/tmp/loop_old_snapshot.ora", &result);
    defer artifacts.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, artifacts.markdown, "assumptions inconsistent") == null);
    try testing.expect(std.mem.indexOf(u8, artifacts.json, "assumptions inconsistent") == null);
    try testing.expect(std.mem.indexOf(u8, artifacts.json, "\"vacuous\":true") == null);
}

test "verification explain mode does not mark valid loop old snapshots inconsistent" {
    const old_entry_source =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total <= std.constants.U256_MAX - 101)
        \\        ensures(total == old(total) + 1)
        \\    {
        \\        total = total + 1;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total == old(total) + 1
        \\        {
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    const loop_entry_source =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total <= std.constants.U256_MAX - 101)
        \\        ensures(total == old(total) + 1)
        \\    {
        \\        total = total + 1;
        \\        let loop_start: u256 = total;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total == loop_start
        \\        {
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    try expectExplainModeNoInconsistentAssumptionNote(old_entry_source);
    try expectExplainModeNoInconsistentAssumptionNote(loop_entry_source);
}

test "verification rejects loop-entry interpretation of old in loop invariants" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total <= std.constants.U256_MAX - 101)
        \\    {
        \\        total = total + 1;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total == old(total)
        \\        {
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification uses invariants for storage-mutating while loops" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total <= std.constants.U256_MAX - 100)
        \\        ensures(total == old(total) + n)
        \\    {
        \\        let start_total: u256 = total;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total == start_total + added
        \\        {
        \\            total = total + 1;
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification preserves pre-loop storage snapshots in effectful loop invariants" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total <= std.constants.U256_MAX - 100)
        \\        ensures(total >= old(total))
        \\    {
        \\        let start_total: u256 = total;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total >= start_total
        \\            invariant total <= start_total + added
        \\        {
        \\            total = total + 1;
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification rejects loop body overflow in inductive iteration" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n == 6)
        \\        requires(total == std.constants.U256_MAX - 5)
        \\        ensures(total >= 0)
        \\    {
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\        {
        \\            total = total + 1;
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(!result.degraded);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification preserves map storage snapshots in effectful loop invariants" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var balances: map<address, u256>;
        \\
        \\    pub fn f(account: address, n: u256)
        \\        requires(n <= 100)
        \\        requires(balances[account] <= std.constants.U256_MAX - 100)
        \\        ensures(balances[account] == old(balances[account]) + n)
        \\    {
        \\        let start_balance: u256 = balances[account];
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant balances[account] == start_balance + added
        \\        {
        \\            balances[account] = balances[account] + 1;
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification preserves multiple storage slots in effectful loop invariants" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract V {
        \\    storage var total_a: u256 = 0;
        \\    storage var total_b: u256 = 0;
        \\
        \\    pub fn f(n: u256)
        \\        requires(n <= 100)
        \\        requires(total_a <= std.constants.U256_MAX - 100)
        \\        requires(total_b <= std.constants.U256_MAX - 100)
        \\        ensures(total_a == old(total_a) + n)
        \\        ensures(total_b == old(total_b) + n)
        \\    {
        \\        let start_a: u256 = total_a;
        \\        let start_b: u256 = total_b;
        \\        var added: u256 = 0;
        \\        while (added < n)
        \\            invariant added <= n
        \\            invariant total_a == start_a + added
        \\            invariant total_b == start_b + added
        \\        {
        \\            total_a = total_a + 1;
        \\            total_b = total_b + 1;
        \\            added = added + 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expect(!result.degraded);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqualStrings("", result.soundness_losses);
}

test "verification infers transitive call-summary slot sorts without degradation" {
    const source_text =
        \\contract Sample {
        \\    storage var value: u256 = 0;
        \\
        \\    fn inner() -> u256 {
        \\        return value;
        \\    }
        \\
        \\    fn outer() -> u256 {
        \\        return inner();
        \\    }
        \\
        \\    pub fn read() -> u256 {
        \\        return outer();
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "read");
    defer result.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports single-field error payload extraction after get_error without degradation" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Sample {
        \\    pub fn inspect(value: Result<bytes, Failure>, flag: bool) -> u256 {
        \\        match (value) {
        \\            Ok(inner) => {
        \\                if (flag) {
        \\                    return 1;
        \\                }
        \\                return 11;
        \\            },
        \\            Err(err) => {
        \\                return err.code;
        \\            }
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "inspect");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports multi-field error payload extraction after get_error without degradation" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\
        \\contract Sample {
        \\    pub fn inspect(value: Result<bytes, Failure>) -> u256 {
        \\        match (value) {
        \\            Ok(_) => {
        \\                return 0;
        \\            },
        \\            Failure(code, owner) => {
        \\                return code;
        \\            }
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "inspect");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct field extraction from known pure callee without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    fn makePair(value: u256) -> Pair {
        \\        return Pair { left: value, right: value };
        \\    }
        \\
        \\    pub fn value_identity(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        let pair = makePair(value);
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification keeps same-named fields on distinct struct sorts independent" {
    const source_text =
        \\struct Debit {
        \\    amount: u256,
        \\}
        \\
        \\struct Toggle {
        \\    amount: bool,
        \\}
        \\
        \\contract Sample {
        \\    pub fn amount_accessors_are_independent(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        let debit = Debit { amount: value };
        \\        let toggle = Toggle { amount: true };
        \\        if (toggle.amount) {
        \\            return debit.amount;
        \\        }
        \\        return 0;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "amount_accessors_are_independent");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports nested named struct extraction from known pure callee without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\struct Wrapper {
        \\    pair: Pair,
        \\    ok: bool,
        \\}
        \\
        \\contract Sample {
        \\    fn wrap(value: u256) -> Wrapper {
        \\        return Wrapper { pair: Pair { left: value, right: value }, ok: true };
        \\    }
        \\
        \\    pub fn value_identity(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        let wrapper = wrap(value);
        \\        return wrapper.pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct extraction through if-returning pure callee without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    fn choose(flag: bool, value: u256) -> Pair {
        \\        if (flag) {
        \\            return Pair { left: value, right: value };
        \\        }
        \\        return Pair { left: value, right: value };
        \\    }
        \\
        \\    pub fn value_identity(flag: bool, value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        let pair = choose(flag, value);
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct extraction after branch-local reassignment without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    pub fn value_identity(flag: bool, value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        var pair = Pair { left: value, right: value };
        \\        if (flag) {
        \\            pair = Pair { left: value, right: value };
        \\        }
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct extraction through switch-returning pure callee without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    fn choose(tag: u8, value: u256) -> Pair {
        \\        switch (tag) {
        \\            0 => return Pair { left: value, right: value };
        \\            1 => return Pair { left: value, right: value };
        \\            else => return Pair { left: value, right: value };
        \\        }
        \\        return Pair { left: value, right: value };
        \\    }
        \\
        \\    pub fn value_identity(tag: u8, value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        let pair = choose(tag, value);
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports tuple extraction from known pure callee without degradation" {
    const source_text =
        \\contract Sample {
        \\    fn makePair(value: u256) -> (u256, u256) {
        \\        return (value, value);
        \\    }
        \\
        \\    pub fn value_identity(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        let pair = makePair(value);
        \\        return pair[1];
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports tuple extraction after branch-local reassignment without degradation" {
    const source_text =
        \\contract Sample {
        \\    pub fn value_identity(flag: bool, value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        var pair: (u256, u256) = (value, value);
        \\        if (flag) {
        \\            pair = (value, value);
        \\        }
        \\        return pair[1];
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports anonymous struct extraction without degradation" {
    const source_text =
        \\contract Sample {
        \\    pub fn value_identity(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        let payload: struct { amount: u256, ok: bool } = .{ .amount = value, .ok = true };
        \\        return payload.amount;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "value_identity");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct field extraction after storage roundtrip without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    storage var saved: Pair;
        \\
        \\    pub fn store_then_read(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        saved = Pair { left: value, right: value };
        \\        let pair: Pair = saved;
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "store_then_read");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct field extraction after storage map roundtrip without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    storage var saved: map<address, Pair>;
        \\
        \\    pub fn store_then_read(account: address, value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        saved[account] = Pair { left: value, right: value };
        \\        let pair: Pair = saved[account];
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "store_then_read");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct field extraction after helper storage write without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    storage var saved: Pair;
        \\
        \\    fn write_saved(value: u256) {
        \\        saved = Pair { left: value, right: value };
        \\    }
        \\
        \\    pub fn store_then_read(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        write_saved(value);
        \\        let pair: Pair = saved;
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "store_then_read");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct field extraction after helper storage map write without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    storage var saved: map<address, Pair>;
        \\
        \\    fn write_saved(account: address, value: u256) {
        \\        saved[account] = Pair { left: value, right: value };
        \\    }
        \\
        \\    pub fn store_then_read(account: address, value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        write_saved(account, value);
        \\        let pair: Pair = saved[account];
        \\        return pair.right;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "store_then_read");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification preserves untouched named struct field after storage field update without degradation" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\
        \\contract Sample {
        \\    storage var saved: Pair;
        \\
        \\    pub fn update_right(left: u256, right: u256) -> u256
        \\        ensures(result == left)
        \\    {
        \\        saved = Pair { left: left, right: 0 };
        \\        saved.right = right;
        \\        let pair: Pair = saved;
        \\        return pair.left;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "update_right");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct field extraction after storage array roundtrip without degradation" {
    const source_text =
        \\struct Point {
        \\    x: u256,
        \\    y: u256,
        \\}
        \\
        \\contract Sample {
        \\    storage var points: [Point; 3];
        \\
        \\    pub fn store_then_read(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        points[1] = Point { x: value, y: value };
        \\        let point: Point = points[1];
        \\        return point.y;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "store_then_read");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named struct field extraction after helper storage array write without degradation" {
    const source_text =
        \\struct Point {
        \\    x: u256,
        \\    y: u256,
        \\}
        \\
        \\contract Sample {
        \\    storage var points: [Point; 3];
        \\
        \\    fn write_point(value: u256) {
        \\        points[1] = Point { x: value, y: value };
        \\    }
        \\
        \\    pub fn store_then_read(value: u256) -> u256
        \\        ensures(result == value)
        \\    {
        \\        write_point(value);
        \\        let point: Point = points[1];
        \\        return point.y;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "store_then_read");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports storage map values containing fixed arrays without degradation" {
    const source_text =
        \\contract MapOfArrays {
        \\    storage var slots: map<address, [u256; 4]>;
        \\
        \\    pub fn set_slot(account: address, index: u256, val: u256) {
        \\        let arr: [u256; 4] = slots[account];
        \\        arr[index] = val;
        \\        slots[account] = arr;
        \\    }
        \\
        \\    pub fn get_slot(account: address, index: u256) -> u256 {
        \\        let arr: [u256; 4] = slots[account];
        \\        return arr[index];
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "set_slot");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification does not vacuously prove branch-local Result unwrap and get_error assumptions" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Sample {
        \\    pub fn inspect(value: Result<bytes, Failure>) -> u256 {
        \\        match (value) {
        \\            Ok(inner) => {
        \\                let _x = inner;
        \\                assert(false);
        \\                return 1;
        \\            },
        \\            Err(err) => {
        \\                return err.code;
        \\            }
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "inspect");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("InvariantViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification uses scf while body condition for callee preconditions" {
    const source_text =
        \\error Rejected;
        \\
        \\contract Sample {
        \\    fn needs_positive(amount: u256) -> !bool | Rejected
        \\        requires(amount > 0)
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn batch(amounts: [u256; 5]) -> u256
        \\        requires(amounts[0] > 0)
        \\        requires(amounts[1] > 0)
        \\        requires(amounts[2] > 0)
        \\        requires(amounts[3] > 0)
        \\        requires(amounts[4] > 0)
        \\    {
        \\        var success_count: u256 = 0;
        \\        var i: u256 = 0;
        \\        while (i < 5)
        \\            invariant i <= 5
        \\            invariant success_count <= i
        \\        {
        \\            try {
        \\                var ok: bool = try needs_positive(amounts[i]);
        \\                if (ok) {
        \\                    success_count += 1;
        \\                }
        \\            } catch (err) {
        \\                let _ignored = err;
        \\            }
        \\            i = i +% 1;
        \\        }
        \\        return success_count;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "batch");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification preserves initialized locals in non-continuing labeled blocks" {
    const source_text =
        \\contract Sample {
        \\    pub fn f() -> u256 {
        \\        var res: u256 = 0;
        \\        outer: {
        \\            assert(res == 0);
        \\            res = res + 1;
        \\            inner: {
        \\                if (res > 10) {
        \\                    break :outer;
        \\                }
        \\                res = res + 1;
        \\                break :inner;
        \\            }
        \\        }
        \\        return res;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification keeps continuing labeled blocks conservative" {
    const source_text =
        \\contract Sample {
        \\    pub fn f() -> u256 {
        \\        var res: u256 = 0;
        \\        outer: {
        \\            res = res + 1;
        \\            if (res < 3) {
        \\                continue :outer;
        \\            }
        \\        }
        \\        return res;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "f");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.errors_len > 0);
    try testing.expect(std.mem.indexOf(u8, result.error_kinds, "InvariantViolation") != null);
    try testing.expect(!result.degraded);
}

test "verification rejects concrete signed min negation overflow" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Sample {
        \\    pub fn negate_min() -> i8 {
        \\        let x: i8 = std.constants.I8_MIN;
        \\        return -x;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "negate_min");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.errors_len > 0);
    try testing.expectEqualStrings("InvariantViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification rejects requires-only unreachable branch obligations" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Sample {
        \\    pub fn g(a: NonZeroAddress) -> bool
        \\        requires a == std.msg.sender()
        \\        ensures a == std.msg.sender()
        \\    {
        \\        let s: NonZeroAddress = std.msg.sender();
        \\
        \\        if (s == a) {
        \\            return true;
        \\        } else {
        \\            let z: u256 = 0;
        \\            let q: u256 = 1 / z;
        \\            return q == 1;
        \\        }
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "g");
    defer result.deinit(testing.allocator);
    try testing.expect(!result.success);
    try testing.expect(result.errors_len > 0);
    try testing.expect(std.mem.indexOf(u8, result.error_kinds, "PreconditionViolation") != null);
    try testing.expect(!result.degraded);
}

test "verification rejects reachable obligations in catch switch and fallthrough paths" {
    const source_text =
        \\error Rejected;
        \\
        \\contract Sample {
        \\    fn maybe(flag: bool) -> !u256 | Rejected {
        \\        if (flag) {
        \\            return Rejected;
        \\        }
        \\        return 1;
        \\    }
        \\
        \\    pub fn catch_path(flag: bool) -> u256 {
        \\        try {
        \\            return try maybe(flag);
        \\        } catch {
        \\            assert(false);
        \\            return 0;
        \\        }
        \\    }
        \\
        \\    pub fn switch_path(tag: u256) -> u256 {
        \\        switch (tag) {
        \\            0 => {
        \\                assert(false);
        \\                return 0;
        \\            }
        \\            else => {
        \\                return 1;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn fallthrough_path(flag: bool) -> u256 {
        \\        if (flag) {
        \\            return 1;
        \\        }
        \\        assert(false);
        \\        return 0;
        \\    }
        \\}
    ;

    var catch_result = try verifyTextWithoutDegradation(source_text, "catch_path");
    defer catch_result.deinit(testing.allocator);
    try testing.expect(!catch_result.success);
    try testing.expectEqualStrings("InvariantViolation", catch_result.error_kinds);
    try testing.expect(!catch_result.degraded);

    var switch_result = try verifyTextWithoutDegradation(source_text, "switch_path");
    defer switch_result.deinit(testing.allocator);
    try testing.expect(!switch_result.success);
    try testing.expectEqualStrings("InvariantViolation", switch_result.error_kinds);
    try testing.expect(!switch_result.degraded);

    var fallthrough_result = try verifyTextWithoutDegradation(source_text, "fallthrough_path");
    defer fallthrough_result.deinit(testing.allocator);
    try testing.expect(!fallthrough_result.success);
    try testing.expectEqualStrings("InvariantViolation", fallthrough_result.error_kinds);
    try testing.expect(!fallthrough_result.degraded);
}

test "verification treats msg.sender as nonzero address" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Sample {
        \\    pub fn fromBuiltinSender() {
        \\        let caller: NonZeroAddress = std.msg.sender();
        \\        assert(caller != std.constants.ZERO_ADDRESS, "refinement fact must hold");
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "fromBuiltinSender");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expect(!result.degraded);
}

test "verification reports invalid refined struct field construction without degradation" {
    const source_text =
        \\struct Box {
        \\    value: MinValue<u256, 10>,
        \\}
        \\
        \\contract Sample {
        \\    pub fn build(raw: u256) -> u256 {
        \\        let box = Box { value: raw };
        \\        return box.value;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "build");
    defer result.deinit(testing.allocator);

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expectEqualStrings("RefinementViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification reports invalid refined ADT payload construction without degradation" {
    const source_text =
        \\enum MaybeAmount {
        \\    None,
        \\    Value(MinValue<u256, 10>),
        \\}
        \\
        \\contract Sample {
        \\    pub fn build(raw: u256) -> u256 {
        \\        let maybe = MaybeAmount.Value(raw);
        \\        return switch (maybe) {
        \\            MaybeAmount.Value(value) => value,
        \\            MaybeAmount.None => 0,
        \\        };
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "build");
    defer result.deinit(testing.allocator);

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expectEqualStrings("RefinementViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification reports invalid MaxValue struct field construction without degradation" {
    const source_text =
        \\struct Box {
        \\    value: MaxValue<u256, 10>,
        \\}
        \\
        \\contract Sample {
        \\    pub fn build(raw: u256) -> u256 {
        \\        let box = Box { value: raw };
        \\        return box.value;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "build");
    defer result.deinit(testing.allocator);

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expectEqualStrings("RefinementViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification reports invalid InRange ADT payload construction without degradation" {
    const source_text =
        \\enum MaybeRate {
        \\    None,
        \\    Value(InRange<u256, 10, 20>),
        \\}
        \\
        \\contract Sample {
        \\    pub fn build(raw: u256) -> u256 {
        \\        let maybe = MaybeRate.Value(raw);
        \\        return switch (maybe) {
        \\            MaybeRate.Value(value) => value,
        \\            MaybeRate.None => 0,
        \\        };
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "build");
    defer result.deinit(testing.allocator);

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expectEqualStrings("RefinementViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification reports invalid NonZeroAddress struct field construction without degradation" {
    const source_text =
        \\struct Holder {
        \\    owner: NonZeroAddress,
        \\}
        \\
        \\contract Sample {
        \\    pub fn build(raw: address) -> address {
        \\        let holder = Holder { owner: raw };
        \\        return holder.owner;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "build");
    defer result.deinit(testing.allocator);

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expectEqualStrings("RefinementViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification reports invalid NonZeroAddress ADT payload construction without degradation" {
    const source_text =
        \\enum MaybeOwner {
        \\    None,
        \\    Value(NonZeroAddress),
        \\}
        \\
        \\contract Sample {
        \\    pub fn build(raw: address) -> address {
        \\        let maybe = MaybeOwner.Value(raw);
        \\        return switch (maybe) {
        \\            MaybeOwner.Value(owner) => owner,
        \\            MaybeOwner.None => raw,
        \\        };
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "build");
    defer result.deinit(testing.allocator);

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expectEqualStrings("RefinementViolation", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification supports multi-error Result match without degradation" {
    const path = "ora-example/corpus/control-flow/match/result_multi_error_match.ora";
    const function_name = "project";

    var result = try verifyExampleWithoutDegradation(path, function_name, false, 5_000);
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expect(!result.degraded);
}

test "verification proves refined struct field extraction without manual assert" {
    const path = "ora-example/corpus/types/refinement/refinement_struct_field_proof.ora";

    var result = try verifyExampleWithoutDegradation(path, "build", false, 5_000);
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 1), result.diagnostics_len);
    try testing.expectEqualStrings("", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification proves refined ADT payload extraction without manual assert" {
    const path = "ora-example/corpus/types/refinement/refinement_adt_payload_proof.ora";

    var result = try verifyExampleWithoutDegradation(path, "build", false, 5_000);
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 1), result.diagnostics_len);
    try testing.expectEqualStrings("", result.error_kinds);
    try testing.expect(!result.degraded);
}

test "verification supports legacy shorthand error union named return without degradation" {
    const source_text =
        \\error ParseError;
        \\
        \\contract Sample {
        \\    fn may_fail(flag: bool) -> !u256 {
        \\        if (flag) {
        \\            return ParseError;
        \\        }
        \\        return 7;
        \\    }
        \\
        \\    pub fn probe() -> u256 {
        \\        return match (may_fail(true)) {
        \\            Ok(_) => 0,
        \\            ParseError => 1,
        \\        };
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "probe");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification distinguishes multi-error variants end-to-end without degradation" {
    const source_text =
        \\error InsufficientBalance;
        \\error Unauthorized;
        \\
        \\contract Sample {
        \\    fn decide(flag: bool) -> !u256 | InsufficientBalance | Unauthorized {
        \\        if (flag) {
        \\            return error.InsufficientBalance();
        \\        }
        \\        return error.Unauthorized();
        \\    }
        \\
        \\    pub fn classify(flag: bool) -> u256
        \\        ensures((flag && result == 1) || (!flag && result == 2))
        \\    {
        \\        return match (decide(flag)) {
        \\            Ok(_) => 0,
        \\            InsufficientBalance => 1,
        \\            Unauthorized => 2,
        \\        };
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "classify");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports symbolic scalar payload enum match without degradation" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\    Pair(u256, u256),
        \\}
        \\
        \\contract PayloadEnumScalarMatch {
        \\    fn choose(flag: bool) -> Event {
        \\        if (flag) {
        \\            return Event.Value(7);
        \\        }
        \\        return Event.Pair(2, 3);
        \\    }
        \\
        \\    pub fn classify(flag: bool) -> u256 {
        \\        let value = switch (choose(flag)) {
        \\            Event.Empty => 0,
        \\            Event.Value(value) => value,
        \\            Event.Pair(lhs, rhs) => lhs,
        \\        };
        \\        if (flag) {
        \\            assert(value == 7, "Value payload should be preserved");
        \\        } else {
        \\            assert(value == 2, "Pair payload should be preserved");
        \\        }
        \\        return value;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "classify");
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports symbolic aggregate payload enum match without degradation" {
    const source_text =
        \\struct Receipt {
        \\    code: u256,
        \\    amount: u256,
        \\}
        \\
        \\enum Event {
        \\    Empty,
        \\    Wrapped(Receipt),
        \\    Named { code: u256, amount: u256 },
        \\}
        \\
        \\contract PayloadEnumAggregateMatch {
        \\    fn choose(flag: bool) -> Event {
        \\        if (flag) {
        \\            let receipt: Receipt = Receipt {
        \\                code: 10,
        \\                amount: 4,
        \\            };
        \\            return Event.Wrapped(receipt);
        \\        }
        \\        return Event.Named(3, 6);
        \\    }
        \\
        \\    pub fn project(flag: bool) -> u256 {
        \\        let value = switch (choose(flag)) {
        \\            Event.Empty => 0,
        \\            Event.Wrapped(receipt) => receipt.code,
        \\            Event.Named(code, amount) => amount,
        \\        };
        \\        if (flag) {
        \\            assert(value == 10, "Wrapped struct payload should be preserved");
        \\        } else {
        \\            assert(value == 6, "Named aggregate payload should be preserved");
        \\        }
        \\        return value;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "project");
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named aggregate payload enum field construction and structural match without degradation" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Named { code: u256, amount: u256 },
        \\}
        \\
        \\contract NamedPayloadStructuralMatch {
        \\    fn choose(flag: bool) -> Event {
        \\        if (flag) {
        \\            return Event.Named {
        \\                amount: 6,
        \\                code: 3,
        \\            };
        \\        }
        \\        return Event.Named {
        \\            amount: 9,
        \\            code: 4,
        \\        };
        \\    }
        \\
        \\    pub fn project(flag: bool) -> u256 {
        \\        let value = switch (choose(flag)) {
        \\            Event.Empty => 0,
        \\            Event.Named { amount: selected, .. } => selected,
        \\        };
        \\        if (flag) {
        \\            assert(value == 6, "field-literal payload should match by name");
        \\        } else {
        \\            assert(value == 9, "structural pattern should bind by name");
        \\        }
        \\        return value;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "project");
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}

test "verification supports named error payload Result match without degradation" {
    const path = "ora-example/corpus/control-flow/match/result_named_error_payload_match.ora";
    const function_name = "project";

    var result = try verifyExampleWithoutDegradation(path, function_name, false, 5_000);
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expect(!result.degraded);
}

test "verification supports named error payload Result roundtrip without degradation" {
    const path = "ora-example/corpus/control-flow/match/result_roundtrip.ora";
    const functions = [_][]const u8{ "make", "consume_and_bump", "project" };

    for (functions) |function_name| {
        var result = try verifyExampleWithoutDegradation(path, function_name, false, 5_000);
        defer result.deinit(testing.allocator);

        try testing.expect(result.success);
        try testing.expectEqual(@as(usize, 0), result.errors_len);
        try testing.expect(!result.degraded);
    }
}

test "verification supports multi-error try propagation without degradation" {
    const path = "ora-example/corpus/types/error/error_propagation.ora";
    const functions = [_][]const u8{ "validate", "compute", "validate_then_compute", "outer" };

    for (functions) |function_name| {
        var result = try verifyExampleWithoutDegradation(path, function_name, false, 5_000);
        defer result.deinit(testing.allocator);

        try testing.expect(result.success);
        try testing.expectEqual(@as(usize, 0), result.errors_len);
        try testing.expect(!result.degraded);
    }
}

test "verification supports wide error-union try catch without degradation" {
    const path = "ora-example/corpus/types/error-union/vault_with_errors.ora";
    const functions = [_][]const u8{ "deposit", "withdraw", "emergency_withdraw_all", "safe_deposit", "safe_withdraw" };

    for (functions) |function_name| {
        var result = try verifyExampleWithoutDegradation(path, function_name, false, 10_000);
        defer result.deinit(testing.allocator);

        try testing.expect(result.success);
        try testing.expectEqual(@as(usize, 0), result.errors_len);
        try testing.expect(!result.degraded);
    }
}

test "verification supports Result is_err on pure helper call without degradation" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\error Failure();
        \\
        \\contract Sample {
        \\    fn choose_u256(flag: bool, value: u256) -> Result<u256, Failure> {
        \\        if (flag) {
        \\            return Ok(value);
        \\        }
        \\        return Err(Failure());
        \\    }
        \\
        \\    pub fn probe(flag: bool, value: u256) -> bool {
        \\        let maybe = choose_u256(flag, value);
        \\        if (!flag) {
        \\            assert(std.result.is_err(maybe));
        \\        }
        \\        return std.result.is_err(maybe);
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "probe");
    defer result.deinit(testing.allocator);
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
    try testing.expectEqual(@as(usize, 0), result.diagnostics_len);
    try testing.expect(!result.degraded);
}
