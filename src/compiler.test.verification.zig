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

test "verification supports multi-error Result match without degradation" {
    const path = "ora-example/corpus/control-flow/match/result_multi_error_match.ora";
    const function_name = "project";

    var result = try verifyExampleWithoutDegradation(path, function_name, false, 5_000);
    defer result.deinit(testing.allocator);

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors_len);
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

