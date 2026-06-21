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

const ORA_BINARY_REL = "zig-out/bin/ora";

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

fn expectSingleUndefinedBogusTypeDiagnostic(source_text: []const u8) !void {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "undefined type 'Bogus'"));
    try testing.expectEqual(@as(usize, 1), countDiagnosticMessages(&typecheck.diagnostics, "undefined type 'Bogus'"));
}

fn expectSingleUndefinedErrorDiagnostic(source_text: []const u8, name: []const u8) !void {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const message = try std.fmt.allocPrint(testing.allocator, "undefined error '{s}'", .{name});
    defer testing.allocator.free(message);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, message));
    try testing.expectEqual(@as(usize, 1), countDiagnosticMessages(&typecheck.diagnostics, message));
}

fn expectSingleBarePipeTypeDiagnostic(source_text: []const u8) !void {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const syntax_diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(syntax_diags, "error-union types must start with '!'"));
    try testing.expectEqual(@as(usize, 1), countDiagnosticMessages(syntax_diags, "error-union types must start with '!'"));
}

fn expectTypecheckOmits(source_text: []const u8, must_not_contain: []const u8) !void {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, must_not_contain));
}

test "compiler accepts contextual short integer literals for address declarations only" {
    const positive =
        \\pub fn zero() -> address {
        \\    let short_zero: address = 0x0;
        \\    let byte_zero: address = 0x00;
        \\    return short_zero;
        \\}
    ;
    var positive_compilation = try compileText(positive);
    defer positive_compilation.deinit();

    const positive_typecheck = try positive_compilation.db.moduleTypeCheck(positive_compilation.root_module_id);
    try testing.expect(positive_typecheck.diagnostics.isEmpty());

    const negative =
        \\pub fn bad(value: u256) -> address {
        \\    return value;
        \\}
    ;
    var negative_compilation = try compileText(negative);
    defer negative_compilation.deinit();

    const negative_typecheck = try negative_compilation.db.moduleTypeCheck(negative_compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&negative_typecheck.diagnostics, "return expects type 'address', found 'u256'"));

    const too_wide =
        \\pub fn bad() -> address {
        \\    let addr: address = 0x10000000000000000000000000000000000000000;
        \\    return addr;
        \\}
    ;
    var too_wide_compilation = try compileText(too_wide);
    defer too_wide_compilation.deinit();

    const too_wide_typecheck = try too_wide_compilation.db.moduleTypeCheck(too_wide_compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&too_wide_typecheck.diagnostics, "does not fit in type 'address'"));
}

// Locks the fail-closed contract for cast-like / overflow builtins: a malformed
// shape MUST be rejected at type-check, never slide through to the HIR lowering
// fallback ("reached HIR lowering without ...") or an ICE. Each needle here was
// confirmed against a freshly built `ora` binary (EXIT=1 at sema, not EXIT=0).
test "cast-like and overflow builtins reject malformed shapes at sema (fail-closed)" {
    // Overflow: wrong arity (binary form needs 2 operands).
    try expectDiagnosticProbeContains(
        \\contract C {
        \\    pub fn f(x: u256) -> u256 {
        \\        let r = @addWithOverflow(x);
        \\        return x;
        \\    }
        \\}
    , .typecheck, "@addWithOverflow expects 2 arguments");

    // Overflow: wrong arity (unary negWithOverflow needs exactly 1 operand).
    try expectDiagnosticProbeContains(
        \\contract C {
        \\    pub fn f(x: u256) -> u256 {
        \\        let r = @negWithOverflow(x, x);
        \\        return x;
        \\    }
        \\}
    , .typecheck, "@negWithOverflow expects 1 arguments");

    // Overflow: non-integer operands.
    try expectDiagnosticProbeContains(
        \\contract C {
        \\    pub fn f() -> u256 {
        \\        let r = @addWithOverflow(true, false);
        \\        return 0;
        \\    }
        \\}
    , .typecheck, "@addWithOverflow expects integer operands");

    // Cast-like: too many value arguments (type slot + 1 value is the only shape).
    try expectDiagnosticProbeContains(
        \\contract C {
        \\    pub fn f(x: u256) -> u256 {
        \\        return @bitCast(u256, x, x);
        \\    }
        \\}
    , .typecheck, "@bitCast expects a type argument and 1 value argument");

    // Cast-like: missing value argument.
    try expectDiagnosticProbeContains(
        \\contract C {
        \\    pub fn f(x: u256) -> u256 {
        \\        return @bitCast(u256);
        \\    }
        \\}
    , .typecheck, "@bitCast expects a type argument and 1 value argument");

    // Cast-like: a non-type in the type slot fails closed via type resolution
    // (@truncate(x) parses `x` as the type argument).
    try expectDiagnosticProbeContains(
        \\contract C {
        \\    pub fn f(x: u256) -> u128 {
        \\        return @truncate(x);
        \\    }
        \\}
    , .typecheck, "undefined type 'x'");

    // Positive control: a well-formed cast must NOT trip the shape check.
    try expectTypecheckOmits(
        \\contract C {
        \\    pub fn f(x: u256) -> u128 {
        \\        return @truncate(u128, x);
        \\    }
        \\}
    , "expects");
}

test "compiler diagnostic release matrix stays readable" {
    try expectDiagnosticProbeContains(
        \\trait Plain {
        \\    fn ping(self) -> bool;
        \\}
        \\
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn badMissingGas(user: address) -> u256 {
        \\        return external<ERC20>(token).balanceOf(user);
        \\    }
        \\}
    ,
        .syntax,
        "expected ', gas: ...' in external proxy",
    );

    try expectDiagnosticProbeContains(
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Nonexistent(7);
        \\}
    ,
        .resolution,
        "undefined name 'Nonexistent'",
    );

    try expectDiagnosticProbeContains(
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Failure(1, 2, 3);
        \\}
    ,
        .typecheck,
        "expected 1 arguments, found 3",
    );

    try expectDiagnosticProbeContains(
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\
        \\    pub fn bad(to: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).transfer(to, balance);
        \\        _ = call_result;
        \\        balance = 2;
        \\    }
        \\}
    ,
        .typecheck,
        "cannot write storage slot 'balance' after external call because it was written before the call",
    );
}

test "compiler reports undefined type names in every annotation position" {
    const cases = [_]struct {
        name: []const u8,
        source: []const u8,
    }{
        .{
            .name = "function parameter",
            .source =
            \\pub fn run(value: Bogus) {
            \\    _ = value;
            \\}
            ,
        },
        .{
            .name = "struct field",
            .source =
            \\struct Holder {
            \\    value: Bogus,
            \\}
            ,
        },
        .{
            .name = "nested struct field",
            .source =
            \\struct Holder {
            \\    value: struct { child: Bogus },
            \\}
            ,
        },
        .{
            .name = "storage var",
            .source =
            \\contract Vault {
            \\    storage var value: Bogus;
            \\}
            ,
        },
        .{
            .name = "map key",
            .source =
            \\contract Vault {
            \\    storage var values: map<Bogus, u256>;
            \\}
            ,
        },
        .{
            .name = "map value",
            .source =
            \\contract Vault {
            \\    storage var values: map<u256, Bogus>;
            \\}
            ,
        },
        .{
            .name = "array element",
            .source =
            \\contract Vault {
            \\    storage var values: [Bogus; 4];
            \\}
            ,
        },
        .{
            .name = "slice element",
            .source =
            \\contract Vault {
            \\    storage var values: slice[Bogus];
            \\}
            ,
        },
        .{
            .name = "tuple member",
            .source =
            \\contract Vault {
            \\    storage var values: (u256, Bogus);
            \\}
            ,
        },
        .{
            .name = "log field",
            .source =
            \\contract Vault {
            \\    log Transfer(value: Bogus);
            \\}
            ,
        },
    };

    for (cases) |case| {
        errdefer std.debug.print("undefined type matrix case failed: {s}\n", .{case.name});
        try expectSingleUndefinedBogusTypeDiagnostic(case.source);
    }
}

test "compiler reports undefined error-union errors once" {
    const cases = [_]struct {
        name: []const u8,
        error_name: []const u8,
        source: []const u8,
    }{
        .{
            .name = "return type",
            .error_name = "BadErr",
            .source =
            \\pub fn run() -> !u256 | BadErr {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "function parameter",
            .error_name = "BadErr",
            .source =
            \\pub fn run(value: !u256 | BadErr) -> u256 {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "one of several",
            .error_name = "BadErr",
            .source =
            \\error Good;
            \\
            \\pub fn run() -> !u256 | Good | BadErr {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "non-error symbol",
            .error_name = "S",
            .source =
            \\struct S {}
            \\
            \\pub fn run() -> !u256 | S {
            \\    return 0;
            \\}
            ,
        },
    };

    for (cases) |case| {
        errdefer std.debug.print("undefined error-union error case failed: {s}\n", .{case.name});
        try expectSingleUndefinedErrorDiagnostic(case.source, case.error_name);
    }
}

test "compiler accepts imported error-union error names" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract StdBytesHelpers {
        \\    pub fn first(data: bytes) -> !u8 | std.bytes.OutOfBounds {
        \\        return std.bytes.at(data, 0);
        \\    }
        \\
        \\    pub fn decodeWord(data: bytes) -> !u256 | std.bytes.InvalidLength {
        \\        return std.bytes.decodeU256BE(data);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler rejects error-union pipes without bang" {
    const cases = [_]struct {
        name: []const u8,
        source: []const u8,
    }{
        .{
            .name = "return type",
            .source =
            \\pub fn run() -> u256 | BadErr {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "function parameter",
            .source =
            \\pub fn run(value: u256 | BadErr) -> u256 {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "defined error still requires bang",
            .source =
            \\error MyErr;
            \\
            \\pub fn run() -> u256 | MyErr {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "non-error symbol",
            .source =
            \\struct S {}
            \\
            \\pub fn run() -> u256 | S {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "numeric suffix",
            .source =
            \\pub fn run() -> u256 | 123 {
            \\    return 0;
            \\}
            ,
        },
        .{
            .name = "empty suffix",
            .source =
            \\pub fn run() -> u256 | {
            \\    return 0;
            \\}
            ,
        },
    };

    for (cases) |case| {
        errdefer std.debug.print("bare pipe type case failed: {s}\n", .{case.name});
        try expectSingleBarePipeTypeDiagnostic(case.source);
    }
}

test "compiler rejects non-name error-union error members" {
    const source_text =
        \\pub fn run() -> !u256 | (u8, u8) {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "error-union error must be a declared error name"));
    try testing.expectEqual(@as(usize, 1), countDiagnosticMessages(&typecheck.diagnostics, "error-union error must be a declared error name"));
}

test "compiler accepts defined error-union errors and preserves lowering" {
    const source_text =
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract Vault {
        \\    pub fn withdraw(amount: u256) -> !u256 | InsufficientBalance {
        \\        return error InsufficientBalance(amount, amount);
        \\    }
        \\}
    ;

    {
        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
        try testing.expect(typecheck.diagnostics.isEmpty());
    }

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @withdraw"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.returns_error_union"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.error_selector"));
}

test "compiler reports unsupported storage Result payload shapes" {
    const dynamic_payload_source =
        \\error Failure;
        \\
        \\contract Vault {
        \\    storage var saved: Result<(u256, u256), Failure>;
        \\}
    ;

    var dynamic_payload = try compileText(dynamic_payload_source);
    defer dynamic_payload.deinit();

    const dynamic_typecheck = try dynamic_payload.db.moduleTypeCheck(dynamic_payload.root_module_id);
    try testing.expect(diagnosticMessagesContain(
        &dynamic_typecheck.diagnostics,
        "storage Result values currently support only scalar, string, bytes, or slice success payloads",
    ));
    try testing.expectEqual(
        @as(usize, 1),
        countDiagnosticMessages(
            &dynamic_typecheck.diagnostics,
            "storage Result values currently support only scalar, string, bytes, or slice success payloads",
        ),
    );

    const error_payload_source =
        \\error Failure(code: u256);
        \\
        \\contract Vault {
        \\    storage var saved: Result<u256, Failure>;
        \\}
    ;

    var error_payload = try compileText(error_payload_source);
    defer error_payload.deinit();

    const error_typecheck = try error_payload.db.moduleTypeCheck(error_payload.root_module_id);
    try testing.expect(diagnosticMessagesContain(
        &error_typecheck.diagnostics,
        "storage Result values currently require payloadless error types",
    ));
    try testing.expectEqual(
        @as(usize, 1),
        countDiagnosticMessages(
            &error_typecheck.diagnostics,
            "storage Result values currently require payloadless error types",
        ),
    );
}

test "compiler rejects opaque computed-storage capability types at unsafe boundaries" {
    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn leak(slot: StorageSlot) {}
        \\}
    , .typecheck, "public function parameter 'slot' cannot expose opaque runtime capability type 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn leak() -> StorageRange {}
        \\}
    , .typecheck, "public function 'leak' cannot expose opaque runtime capability return type 'StorageRange'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    storage var slot: StorageSlot;
        \\}
    , .typecheck, "storage declarations cannot use opaque storage capability type 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    storage var indexed: map<StorageSlot, u256>;
        \\}
    , .typecheck, "storage declarations cannot use opaque storage capability type 'map<StorageSlot, u256>'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(raw: u256) {
        \\        let slot: StorageSlot = @cast(StorageSlot, raw);
        \\    }
        \\}
    , .typecheck, "@cast cannot construct opaque runtime capability type 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(raw: u256) {
        \\        let slot: StorageSlot = @bitCast(StorageSlot, raw);
        \\    }
        \\}
    , .typecheck, "@bitCast cannot construct opaque runtime capability type 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(owner: address) -> u256 {
        \\        let slot: StorageSlot = @storageDerive("vault.payload", owner);
        \\        return @cast(u256, slot);
        \\    }
        \\}
    , .typecheck, "@cast cannot reinterpret opaque runtime capability type 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(owner: address) -> u256 {
        \\        let slot: StorageSlot = @storageDerive("vault.payload", owner);
        \\        return @truncate(u256, slot);
        \\    }
        \\}
    , .typecheck, "@truncate cannot reinterpret opaque runtime capability type 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(owner: address) -> u256 {
        \\        let slot: StorageSlot = @storageDerive("vault.payload", owner);
        \\        let next = slot + 1;
        \\        return 0;
        \\    }
        \\}
    , .typecheck, "invalid binary operator '+' for types 'StorageSlot' and 'integer'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(owner: address) -> bool {
        \\        let slot: StorageSlot = @storageDerive("vault.payload", owner);
        \\        return slot == slot;
        \\    }
        \\}
    , .typecheck, "invalid binary operator '==' for types 'StorageSlot' and 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(owner: address) {
        \\        let ns = "vault.dynamic";
        \\        let slot: StorageSlot = @storageDerive(ns, owner);
        \\    }
        \\}
    , .typecheck, "@storageDerive namespace must be a string literal");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(key: string) {
        \\        let slot: StorageSlot = @storageDerive("vault.dynamic", key);
        \\    }
        \\}
    , .typecheck, "@storageDerive key type 'string' is not supported");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(raw: u256) -> u256 {
        \\        return @storageWordLoad(raw, 0);
        \\    }
        \\}
    , .typecheck, "@storageWordLoad expects StorageSlot as its first argument, found 'u256'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(raw: u256) {
        \\        @storageWordStore(raw, 0, 1);
        \\    }
        \\}
    , .typecheck, "@storageWordStore expects StorageSlot as its first argument, found 'u256'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(raw: u256) {
        \\        let range: StorageRange = @storageRange(raw, 2);
        \\    }
        \\}
    , .typecheck, "@storageRange expects StorageSlot as its first argument, found 'u256'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(raw: u256) {
        \\        @storageRangeErase(raw);
        \\    }
        \\}
    , .typecheck, "@storageRangeErase expects StorageRange, found 'u256'");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    pub fn bad(owner: address, len: u256) {
        \\        let slot: StorageSlot = @storageDerive("vault.dynamic", owner);
        \\        let range: StorageRange = @storageRange(slot, len);
        \\        @storageRangeErase(range);
        \\    }
        \\}
    , .typecheck, "@storageRange length must be a compile-time integer literal or comptime integer parameter");

    try expectDiagnosticProbeContains(
        \\comptime const std_storage = @import("std/storage");
        \\
        \\contract Vault {
        \\    pub fn bad(owner: address, len: u256) {
        \\        let slot: StorageSlot = std_storage.derive("vault.dynamic", owner);
        \\        let range: StorageRange = std_storage.range(slot, len);
        \\        @storageRangeErase(range);
        \\    }
        \\}
    , .typecheck, "std.storage.range length must be a compile-time integer literal or comptime integer parameter");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    struct Payload { value: u256 }
        \\
        \\    pub fn bad(owner: address, payload: Payload) {
        \\        let slot: StorageSlot = @storageDerive("vault.payload", owner);
        \\        @storageWordStore(slot, 0, payload);
        \\    }
        \\}
    , .typecheck, "@storageWordStore value must be an integer-compatible word");
}

test "compiler accepts resource domains and storage resource places" {
    const source_text =
        \\resource TokenUnit = u256;
        \\resource ShareUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\    storage var reserve: Resource<TokenUnit>;
        \\    log Transfer(to: address, amount: TokenUnit);
        \\
        \\    pub fn identity(amount: TokenUnit) -> TokenUnit {
        \\        return amount;
        \\    }
        \\
        \\    pub fn literal() -> TokenUnit {
        \\        return 10;
        \\    }
        \\
        \\    pub fn balanceOf(owner: address) -> TokenUnit {
        \\        return balances[owner];
        \\    }
        \\
        \\    pub fn reserveBalance() -> TokenUnit {
        \\        return reserve;
        \\    }
        \\
        \\    fn addSameDomain(lhs: TokenUnit, rhs: TokenUnit) -> TokenUnit {
        \\        let one: TokenUnit = 1;
        \\        return lhs + rhs + one;
        \\    }
        \\
        \\    pub fn announce(to: address, amount: TokenUnit) {
        \\        log Transfer(to, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler rejects invalid resource declarations and first-class resource places" {
    try expectDiagnosticProbeContains(
        \\resource TokenUnit = bool;
    , .typecheck, "resource carrier for 'TokenUnit' must be an integer type, found 'bool'");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = TokenUnit;
    , .typecheck, "recursive resource declaration 'TokenUnit' is not supported");

    try expectDiagnosticProbeContains(
        \\contract Vault {
        \\    storage var balances: map<address, Resource<u256>>;
        \\}
    , .typecheck, "Resource<T> expects a resource-domain type argument, found 'u256'");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<Resource<TokenUnit>>>;
        \\}
    , .typecheck, "Resource<T> expects a resource-domain type argument, found 'Resource<TokenUnit>'");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    pub fn expose(place: Resource<TokenUnit>) {}
        \\}
    , .typecheck, "public function parameter 'place' cannot expose opaque runtime capability type 'Resource<TokenUnit>'");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    fn local(place: Resource<TokenUnit>) {}
        \\}
    , .typecheck, "function parameter 'place' cannot have resource place type 'Resource<TokenUnit>' as a runtime value");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    log Bad(place: Resource<TokenUnit>);
        \\}
    , .typecheck, "log field 'place' cannot expose opaque runtime capability type 'Resource<TokenUnit>'");

    try expectDiagnosticProbeContains(
        \\resource USDC = u256;
        \\resource DAI = u256;
        \\
        \\fn bad(lhs: USDC, rhs: DAI) -> USDC {
        \\    return lhs + rhs;
        \\}
    , .typecheck, "invalid binary operator '+' for types 'USDC' and 'DAI'");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\fn bad(lhs: TokenUnit, rhs: u256) -> TokenUnit {
        \\    return lhs + rhs;
        \\}
    , .typecheck, "invalid binary operator '+' for types 'TokenUnit' and 'u256'");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\fn bad(raw: u256) -> TokenUnit {
        \\    let amount: TokenUnit = raw;
        \\    return amount;
        \\}
    , .typecheck, "declaration expects type 'TokenUnit', found 'u256'");
}

test "compiler validates resource move create and destroy builtins" {
    const source_text =
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn transfer(from: address, to: address, amount: TokenUnit) {
        \\        @move(balances[from], balances[to], amount);
        \\    }
        \\
        \\    pub fn issue(to: address, amount: TokenUnit) {
        \\        @create(balances[to], amount);
        \\    }
        \\
        \\    pub fn retire(from: address, amount: TokenUnit) {
        \\        @destroy(balances[from], amount);
        \\    }
        \\
        \\    pub fn balanceOf(owner: address) -> TokenUnit {
        \\        return balances[owner];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler rejects invalid resource builtin calls" {
    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(from: address, to: address) {
        \\        @move(balances[from], balances[to]);
        \\    }
        \\}
    , .typecheck, "@move expects 3 arguments");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(to: address, amount: TokenUnit) {
        \\        @move(amount, balances[to], amount);
        \\    }
        \\}
    , .typecheck, "@move expects Resource<T> places as its first two arguments");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    pub fn bad(amount: TokenUnit) {
        \\        @create(amount, amount);
        \\    }
        \\}
    , .typecheck, "@create expects a Resource<T> place as its first argument");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    pub fn bad(amount: TokenUnit) {
        \\        @destroy(amount, amount);
        \\    }
        \\}
    , .typecheck, "@destroy expects a Resource<T> place as its first argument");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\struct Bucket {
        \\    balance: Resource<TokenUnit>;
        \\}
        \\
        \\contract Vault {
        \\    storage var buckets: map<address, Bucket>;
        \\
        \\    pub fn bad(owner: address, amount: TokenUnit) {
        \\        @create(buckets[owner].balance, amount);
        \\    }
        \\}
    , .typecheck, "resource struct-field places inside maps are not supported yet; use a direct storage Resource<T> root, direct struct field, or map value resource place");

    try expectDiagnosticProbeContains(
        \\resource USDC = u256;
        \\resource DAI = u256;
        \\
        \\contract Vault {
        \\    storage var usdc: map<address, Resource<USDC>>;
        \\    storage var dai: map<address, Resource<DAI>>;
        \\
        \\    pub fn bad(from: address, to: address, amount: USDC) {
        \\        @move(usdc[from], dai[to], amount);
        \\    }
        \\}
    , .typecheck, "cannot move between Resource<USDC> and Resource<DAI>");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(from: address, to: address, amount: u256) {
        \\        @move(balances[from], balances[to], amount);
        \\    }
        \\}
    , .typecheck, "resource amount must have type TokenUnit");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    fn dynamicKey() -> address {
        \\        return msg.sender;
        \\    }
        \\
        \\    pub fn bad(to: address, amount: TokenUnit) {
        \\        @move(balances[dynamicKey()], balances[to], amount);
        \\    }
        \\}
    , .typecheck, "resource place key expression must be side-effect-free");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var nested: map<address, map<address, Resource<TokenUnit>>>;
        \\
        \\    fn dynamicKey() -> address {
        \\        return msg.sender;
        \\    }
        \\
        \\    pub fn bad(to: address, amount: TokenUnit) {
        \\        @create(nested[dynamicKey()][to], amount);
        \\    }
        \\}
    , .typecheck, "resource place key expression must be side-effect-free");

    try expectDiagnosticProbeContains(
        \\resource DebtUnit = i256;
        \\
        \\contract Vault {
        \\    storage var debts: map<address, Resource<DebtUnit>>;
        \\
        \\    pub fn bad(to: address) {
        \\        @create(debts[to], -1);
        \\    }
        \\}
    , .typecheck, "resource amount must be non-negative");
}

test "compiler rejects direct mutation of resource places" {
    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(to: address, amount: TokenUnit) {
        \\        balances[to] = amount;
        \\    }
        \\}
    , .typecheck, "resource places can only be mutated with @move, @create, or @destroy");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(to: address, amount: TokenUnit) {
        \\        balances[to] += amount;
        \\    }
        \\}
    , .typecheck, "resource places can only be mutated with @move, @create, or @destroy");
}

test "compiler rejects resource boundary builtins in value positions" {
    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(from: address, to: address, amount: TokenUnit) {
        \\        let output = @move(balances[from], balances[to], amount);
        \\    }
        \\}
    , .typecheck, "@move is statement-only and cannot be used in expression position");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(to: address, amount: TokenUnit) {
        \\        let output = @create(balances[to], amount);
        \\    }
        \\}
    , .typecheck, "@create is statement-only and cannot be used in expression position");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(from: address, amount: TokenUnit) {
        \\        let output = @destroy(balances[from], amount);
        \\    }
        \\}
    , .typecheck, "@destroy is statement-only and cannot be used in expression position");
}

test "compiler tracks resource builtin effects for modifies and locks" {
    const covered_source =
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn transfer(from: address, to: address, amount: TokenUnit)
        \\        modifies balances[from], balances[to]
        \\    {
        \\        @move(balances[from], balances[to], amount);
        \\    }
        \\}
    ;
    var covered = try compileText(covered_source);
    defer covered.deinit();
    const covered_typecheck = try covered.db.moduleTypeCheck(covered.root_module_id);
    try testing.expect(covered_typecheck.diagnostics.isEmpty());

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn transfer(from: address, to: address, amount: TokenUnit)
        \\        modifies balances[from]
        \\    {
        \\        @move(balances[from], balances[to], amount);
        \\    }
        \\}
    , .typecheck, "is not covered by this function's `modifies` clause");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn bad(user: address, amount: TokenUnit) {
        \\        @lock(balances[user]);
        \\        @create(balances[user], amount);
        \\    }
        \\}
    , .typecheck, "cannot write locked storage slot 'balances'");

    const direct_storage_source =
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var reserve: Resource<TokenUnit>;
        \\    storage var treasury: Resource<TokenUnit>;
        \\
        \\    pub fn sweep(amount: TokenUnit)
        \\        modifies reserve, treasury
        \\    {
        \\        @move(reserve, treasury, amount);
        \\    }
        \\}
    ;
    var direct_storage = try compileText(direct_storage_source);
    defer direct_storage.deinit();
    const direct_storage_typecheck = try direct_storage.db.moduleTypeCheck(direct_storage.root_module_id);
    try testing.expect(direct_storage_typecheck.diagnostics.isEmpty());

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var reserve: Resource<TokenUnit>;
        \\    storage var treasury: Resource<TokenUnit>;
        \\
        \\    pub fn sweep(amount: TokenUnit)
        \\        modifies reserve
        \\    {
        \\        @move(reserve, treasury, amount);
        \\    }
        \\}
    , .typecheck, "is not covered by this function's `modifies` clause");

    try expectDiagnosticProbeContains(
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var reserve: Resource<TokenUnit>;
        \\
        \\    pub fn bad(amount: TokenUnit) {
        \\        @lock(reserve);
        \\        @destroy(reserve, amount);
        \\    }
        \\}
    , .typecheck, "cannot write locked storage slot 'reserve'");
}

test "compiler build rejects unbounded computed storage ranges without artifacts" {
    std.fs.cwd().access(ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const std_storage = @import("std/storage");
        \\
        \\contract Vault {
        \\    pub fn bad(owner: address, len: u256, value: u256)
        \\        modifies std_storage.range(std_storage.derive("vault.unbounded", owner), len)
        \\    {
        \\        let slot: StorageSlot = std_storage.derive("vault.unbounded", owner);
        \\        std_storage.words.store(slot, 0, value);
        \\    }
        \\}
        ,
    });
    try tmp.dir.makeDir("out");

    const root_path = try pathFromTmpAlloc(testing.allocator, tmp, "main.ora");
    defer testing.allocator.free(root_path);
    const out_path = try pathFromTmpAlloc(testing.allocator, tmp, "out");
    defer testing.allocator.free(out_path);

    const result = try std.process.Child.run(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{
            ORA_BINARY_REL,
            "build",
            "-o",
            out_path,
            root_path,
        },
        .max_output_bytes = 1024 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| try testing.expect(code != 0),
        else => return error.TestUnexpectedResult,
    }
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "`modifies` computed storage ranges require a literal bounded word count in v1") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "`modifies` computed storage ranges require a literal bounded word count in v1"));
    try testing.expectError(error.FileNotFound, tmp.dir.access("out/bin/main.hex", .{}));
}

test "compiler abi emit barrier rejects undefined types without artifacts" {
    std.fs.cwd().access(ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\contract Vault {
        \\    storage var value: Bogus;
        \\
        \\    pub fn run() -> u256 {
        \\        return 0;
        \\    }
        \\}
        ,
    });
    try tmp.dir.makeDir("out");

    const root_path = try pathFromTmpAlloc(testing.allocator, tmp, "main.ora");
    defer testing.allocator.free(root_path);
    const out_path = try pathFromTmpAlloc(testing.allocator, tmp, "out");
    defer testing.allocator.free(out_path);

    const result = try std.process.Child.run(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{
            ORA_BINARY_REL,
            "emit",
            "--emit=abi:extras",
            "-o",
            out_path,
            root_path,
        },
        .max_output_bytes = 1024 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| try testing.expect(code != 0),
        else => return error.TestUnexpectedResult,
    }
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "undefined type 'Bogus'") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "undefined type 'Bogus'"));

    var out_dir = try tmp.dir.openDir("out", .{ .iterate = true });
    defer out_dir.close();
    var iter = out_dir.iterate();
    try testing.expect((try iter.next()) == null);
}

test "compiler abi emit barrier rejects undefined error-union errors without artifacts" {
    std.fs.cwd().access(ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\contract Vault {
        \\    pub fn run() -> !u256 | BadErr {
        \\        return 0;
        \\    }
        \\}
        ,
    });
    try tmp.dir.makeDir("out");

    const root_path = try pathFromTmpAlloc(testing.allocator, tmp, "main.ora");
    defer testing.allocator.free(root_path);
    const out_path = try pathFromTmpAlloc(testing.allocator, tmp, "out");
    defer testing.allocator.free(out_path);

    const result = try std.process.Child.run(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{
            ORA_BINARY_REL,
            "emit",
            "--emit=abi",
            "-o",
            out_path,
            root_path,
        },
        .max_output_bytes = 1024 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| try testing.expect(code != 0),
        else => return error.TestUnexpectedResult,
    }
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "undefined error 'BadErr'") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "undefined error 'BadErr'"));

    var out_dir = try tmp.dir.openDir("out", .{ .iterate = true });
    defer out_dir.close();
    var iter = out_dir.iterate();
    try testing.expect((try iter.next()) == null);
}

test "compiler abi emit barrier rejects error-union pipes without bang without artifacts" {
    std.fs.cwd().access(ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\contract Vault {
        \\    pub fn run() -> u256 | BadErr {
        \\        return 0;
        \\    }
        \\}
        ,
    });
    try tmp.dir.makeDir("out");

    const root_path = try pathFromTmpAlloc(testing.allocator, tmp, "main.ora");
    defer testing.allocator.free(root_path);
    const out_path = try pathFromTmpAlloc(testing.allocator, tmp, "out");
    defer testing.allocator.free(out_path);

    const result = try std.process.Child.run(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{
            ORA_BINARY_REL,
            "emit",
            "--emit=abi",
            "-o",
            out_path,
            root_path,
        },
        .max_output_bytes = 1024 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| try testing.expect(code != 0),
        else => return error.TestUnexpectedResult,
    }
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "error-union types must start with '!'") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "error-union types must start with '!'"));

    var out_dir = try tmp.dir.openDir("out", .{ .iterate = true });
    defer out_dir.close();
    var iter = out_dir.iterate();
    try testing.expect((try iter.next()) == null);
}

test "compiler build rejects unsupported local Result aggregate carriers before bytecode artifacts" {
    std.fs.cwd().access(ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\error Failure;
        \\
        \\contract ResultStringMemRefDead {
        \\    pub fn run() -> u256 {
        \\        var values: [Result<string, Failure>; 1] = [Err(Failure())];
        \\        values[0] = Ok("abc");
        \\        return 7;
        \\    }
        \\}
        ,
    });
    try tmp.dir.makeDir("out");

    const root_path = try pathFromTmpAlloc(testing.allocator, tmp, "main.ora");
    defer testing.allocator.free(root_path);
    const out_path = try pathFromTmpAlloc(testing.allocator, tmp, "out");
    defer testing.allocator.free(out_path);

    const result = try std.process.Child.run(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{
            ORA_BINARY_REL,
            "build",
            "-o",
            out_path,
            root_path,
        },
        .max_output_bytes = 1024 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| try testing.expect(code != 0),
        else => return error.TestUnexpectedResult,
    }
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "local Result aggregate values currently require a scalar success payload") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "local Result aggregate values currently require a scalar success payload"));

    try testing.expectError(error.FileNotFound, tmp.dir.access("out/bin/main.hex", .{}));
}

test "compiler build removes partial artifacts when native SIR lowering fails" {
    std.fs.cwd().access(ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\contract Entry {
        \\    pub fn take_nested(values: slice[slice[u256]]) -> bool {
        \\        return true;
        \\    }
        \\}
        ,
    });

    const root_path = try pathFromTmpAlloc(testing.allocator, tmp, "main.ora");
    defer testing.allocator.free(root_path);
    const out_path = try pathFromTmpAlloc(testing.allocator, tmp, "out");
    defer testing.allocator.free(out_path);

    const result = try std.process.Child.run(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{
            ORA_BINARY_REL,
            "build",
            "-o",
            out_path,
            root_path,
        },
        .max_output_bytes = 1024 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| try testing.expect(code != 0),
        else => return error.TestUnexpectedResult,
    }
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "unsupported dynamic ABI type for dispatcher") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "unsupported dynamic ABI type for dispatcher"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "SIR dispatcher build failed") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "SIR dispatcher build failed"));

    try tmp.dir.access("out", .{});
    try testing.expectError(error.FileNotFound, tmp.dir.access("out/bin/main.hex", .{}));
}

test "compiler build accepts public fallible returns with no declared custom errors" {
    std.fs.cwd().access(ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\contract Entry {
        \\    pub fn ok() -> !bool {
        \\        return true;
        \\    }
        \\}
        ,
    });

    const root_path = try pathFromTmpAlloc(testing.allocator, tmp, "main.ora");
    defer testing.allocator.free(root_path);
    const out_path = try pathFromTmpAlloc(testing.allocator, tmp, "out");
    defer testing.allocator.free(out_path);

    const result = try std.process.Child.run(.{
        .allocator = testing.allocator,
        .argv = &[_][]const u8{
            ORA_BINARY_REL,
            "build",
            "-o",
            out_path,
            root_path,
        },
        .max_output_bytes = 1024 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| try testing.expectEqual(@as(u8, 0), code),
        else => return error.TestUnexpectedResult,
    }
    try tmp.dir.access("out/bin/main.hex", .{});
}

test "compiler reports undefined type names at value resolution positions once" {
    const cases = [_]struct {
        name: []const u8,
        source: []const u8,
    }{
        .{
            .name = "return type",
            .source =
            \\pub fn run() -> Bogus {
            \\    return 1;
            \\}
            ,
        },
        .{
            .name = "typed local initializer",
            .source =
            \\pub fn run() {
            \\    let value: Bogus = 1;
            \\    _ = value;
            \\}
            ,
        },
        .{
            .name = "cast target",
            .source =
            \\pub fn run(value: u256) -> u256 {
            \\    return @cast(Bogus, value);
            \\}
            ,
        },
    };

    for (cases) |case| {
        errdefer std.debug.print("undefined type value-position case failed: {s}\n", .{case.name});
        try expectSingleUndefinedBogusTypeDiagnostic(case.source);
    }
}

test "compiler treats result as an ensures-only reserved pseudo variable" {
    {
        const source_text =
            \\pub fn ok(value: u256) -> u256
            \\    ensures(result >= value)
            \\{
            \\    return value;
            \\}
        ;

        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const resolution_diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
        try testing.expectEqual(@as(usize, 0), resolution_diags.len());
    }

    try expectDiagnosticProbeContains(
        \\pub fn bad(value: u256) -> u256
        \\    requires(result >= value)
        \\{
        \\    return value;
        \\}
    ,
        .resolution,
        "undefined name 'result'",
    );

    {
        const source_text =
            \\pub fn bad() -> u256 {
            \\    let result = 1;
            \\    return 0;
            \\}
        ;

        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const module = compilation.db.sources.module(compilation.root_module_id);
        const ast_diags = try compilation.db.astDiagnostics(module.file_id);
        try testing.expect(diagnosticMessagesContain(ast_diags, "'result' is a reserved keyword and cannot be used as a variable name"));
    }
}

test "compiler rejects payloadless named error arms that bind a payload" {
    const source_text =
        \\error Failure;
        \\error Denied;
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Failure(err) => 7,
        \\        Denied => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "payloadless named error match arms cannot bind a payload; use 'Failure =>' instead"));
}

test "compiler rejects multi-field named error payload bindings" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\error Denied;
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Failure(err) => 7,
        \\        Denied => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "named error payload bindings must match the error payload field count"));
}

test "compiler rejects unknown error returns" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Nonexistent(7);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const resolution_diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(resolution_diags, "undefined name 'Nonexistent'"));
}

test "compiler rejects error returns with wrong payload arity" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Failure(1, 2, 3);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "expected 1 arguments, found 3"));
}

test "compiler rejects error returns outside function return error set" {
    const source_text =
        \\error ErrorA;
        \\error ErrorB;
        \\
        \\pub fn run() -> !u256 | ErrorA {
        \\    return ErrorB();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "return expects type '!u256 | ErrorA', found 'ErrorB'"));
}

test "compiler rejects error returns with wrong payload types" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Failure(true, 7);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.items.items.len != 0);
}

test "compiler rejects mixed explicit and implicit string enum values" {
    const source_text =
        \\enum ErrorCode : string {
        \\    InvalidInput = "ERR_INVALID_INPUT",
        \\    Legacy,
        \\    Unauthorized = "ERR_UNAUTHORIZED",
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "string enums with explicit values must assign every variant explicitly"));
}

test "compiler rejects bytes enums without explicit values for every variant" {
    const source_text =
        \\enum Signature : bytes {
        \\    A = hex"deadbeef",
        \\    B,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "bytes enums currently require explicit values for every variant"));
}

test "compiler rejects unsupported integer type widths" {
    try expectDiagnosticProbeContains(
        \\pub fn bad(value: u24) -> u24 {
        \\    return value;
        \\}
    ,
        .typecheck,
        "invalid integer type 'u24'; supported integer types are u8, u16, u32, u64, u128, u160, u256, i8, i16, i32, i64, i128, and i256",
    );

    try expectDiagnosticProbeContains(
        \\pub fn bad(value: i96) -> i96 {
        \\    return value;
        \\}
    ,
        .typecheck,
        "invalid integer type 'i96'",
    );
}

test "compiler parses enum explicit value expressions without syntax diagnostics" {
    const source_text =
        \\contract T {
        \\    enum TestEnum {
        \\        Value1 = 5,
        \\        Value2 = 10 + 20,
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expectEqual(@as(usize, 0), diags.items.items.len);
}

test "compiler rejects non-comptime enum explicit value expressions" {
    const source_text =
        \\storage var seed: u256;
        \\
        \\enum Bad {
        \\    A = seed + 1,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "explicit values for integer enums must be compile-time integer expressions"));
}

test "compiler reports impl syntax errors for missing body and missing for" {
    const source_text =
        \\impl ERC20 Token {
        \\    fn totalSupply(self) -> u256;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(diags, "expected 'for' in impl declaration"));
    try testing.expect(diagnosticMessagesContain(diags, "impl methods must have a body"));
}

test "compiler rejects bare self in ordinary functions" {
    const source_text =
        \\pub fn bad(self) -> u256 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(ast_diags, "bare 'self' parameter is only allowed in trait and impl methods"));
}

test "compiler rejects single-element tuple type syntax instead of unwrapping it" {
    const source_text =
        \\type Wrapped = (u256,);
        \\pub fn f(value: (u256,)) -> (u256,) {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(ast_diags, "single-element tuple types are not supported; use the element type directly"));
}

test "compiler accepts parenthesized type syntax without treating it as a one-tuple" {
    const source_text =
        \\type Items = (slice[u256]);
        \\type Bound = (MinValue<u256, 1>);
        \\type Pair = (u256, bool,);
        \\pub fn f(value: (Items)) -> u256 {
        \\    return 0;
        \\}
        \\pub fn g(value: Bound, pair: Pair) -> u256 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler rejects negative refinement bounds on unsigned integer bases" {
    const source_text =
        \\type BadMin = MinValue<u8, -5>;
        \\type BadMax = MaxValue<u8, -100>;
        \\type BadRange = InRange<u8, -1, 10>;
        \\pub fn f(value: BadMin, max: BadMax, other: BadRange) -> u8 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "MinValue bound -5 is negative; u8 base requires non-negative bound"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "MaxValue bound -100 is negative; u8 base requires non-negative bound"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "InRange bound -1 is negative; u8 base requires non-negative bound"));
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

test "compiler reports heterogeneous array literals and keeps tuple element types" {
    const source_text =
        \\pub fn build(flag: bool, small: u8, big: u256) -> bool {
        \\    let ok: [u256; 3] = [1, 2, 3];
        \\    let widened: [u256; 3] = [small, big, 3];
        \\    let bad = [big, false];
        \\    let pair: (bool, u256) = (flag, 7);
        \\    return pair[0];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 1), type_diags.len());
    try testing.expectEqualStrings("array literal elements have incompatible types 'u256' and 'bool'", type_diags.items.items[0].message);

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

test "compiler rejects first-class runtime map handles" {
    const source_text =
        \\contract NestedMaps {
        \\    storage var allowances: map<address, map<address, u256>>;
        \\
        \\    pub fn inferred(owner: address, spender: address, amount: u256) {
        \\        let inner = allowances[owner];
        \\        inner[spender] = amount;
        \\    }
        \\
        \\    pub fn explicit(owner: address, spender: address, amount: u256) {
        \\        let inner: map<address, u256> = allowances[owner];
        \\        inner[spender] = amount;
        \\    }
        \\
        \\    fn helper(inner: map<address, u256>, spender: address, amount: u256) {
        \\        inner[spender] = amount;
        \\    }
        \\
        \\    fn returns(owner: address) -> map<address, u256> {
        \\        return allowances[owner];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const local_message = "local 'inner' cannot have map type 'map<address, u256>' as a runtime value; index storage maps directly";
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, local_message));
    try testing.expectEqual(@as(usize, 2), countDiagnosticMessages(&typecheck.diagnostics, local_message));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "function parameter 'inner' cannot have map type 'map<address, u256>' as a runtime value; index storage maps directly"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "function 'returns' cannot return map type 'map<address, u256>'; maps are storage roots, not first-class runtime values"));
}

test "compiler rejects integer array literals assigned to bool arrays" {
    const source_text =
        \\pub fn build() -> [bool; 2] {
        \\    let dest: [bool; 2] = [0, 0];
        \\    return dest;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "declaration expects type '[bool; 2]', found '[integer; 2]'"));
}

test "compiler rejects log statements with wrong arity" {
    const source_text =
        \\contract C {
        \\    log Transfer(from: address, amount: u256);
        \\
        \\    pub fn run(addr: address) {
        \\        log Transfer(addr);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "log 'Transfer' expects 2 arguments, found 1"));
}

test "compiler rejects log declarations with more than three indexed fields" {
    const source_text =
        \\contract C {
        \\    log TooMany(indexed a: address, indexed b: address, indexed c: address, indexed d: address);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "log declarations support at most 3 indexed fields"));
}

test "compiler rejects struct-typed indexed log fields" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\contract C {
        \\    log Bad(indexed p: Pair);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "indexed log field 'p' has unsupported type 'Pair'"));
}

test "compiler rejects dynamic indexed log fields until topic hashing is supported" {
    const source_text =
        \\contract C {
        \\    log Bad(indexed label: string);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "indexed log field 'label' has unsupported type 'string'"));
}

test "compiler rejects opaque storage capability log fields" {
    try expectDiagnosticProbeContains(
        \\contract C {
        \\    log Bad(slot: StorageSlot);
        \\}
    , .typecheck, "log field 'slot' cannot expose opaque runtime capability type 'StorageSlot'");

    try expectDiagnosticProbeContains(
        \\contract C {
        \\    log Bad(indexed range: StorageRange);
        \\}
    , .typecheck, "log field 'range' cannot expose opaque runtime capability type 'StorageRange'");
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
    try testing.expectEqual(@as(usize, 5), countDiagnosticMessages(type_diags, "constant value 256 does not fit in type 'u8'"));
}

test "compiler reports signed integer constant overflow at 256-bit boundaries" {
    const signed_source =
        \\pub fn signed_too_large() -> i256 {
        \\    let value: i256 = 0x8000000000000000000000000000000000000000000000000000000000000000;
        \\    return value;
        \\}
        \\
        \\pub fn signed_too_small() -> i256 {
        \\    let value: i256 = -57896044618658097711785492504343953926634992332820282019728792003956564819969;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(signed_source);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const too_large_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const too_small_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expect(diagnosticMessagesContain(too_large_diags, "does not fit in type 'i256'"));
    try testing.expect(diagnosticMessagesContain(too_small_diags, "does not fit in type 'i256'"));
}

test "compiler reports integer constant overflow at call sites" {
    const source_text =
        \\fn take(value: u8) -> u8 {
        \\    return value;
        \\}
        \\
        \\pub fn narrow() -> u8 {
        \\    return take(256);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expect(countDiagnosticMessages(type_diags, "constant value 256 does not fit in type 'u8'") >= 1);
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

test "compiler resolves comptime integer operands through concrete integer contexts" {
    const source_text =
        \\pub fn narrow(x: u8) -> u8 {
        \\    let a: u8 = 1 + 2;
        \\    let b = x + 3;
        \\    let c = 4 + x;
        \\    return a + b + c;
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
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const a_pattern = findVariablePatternByName(ast_file, body.statements, "a").?;
    const b_pattern = findVariablePatternByName(ast_file, body.statements, "b").?;
    const c_pattern = findVariablePatternByName(ast_file, body.statements, "c").?;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[a_pattern.index()].kind());
    try testing.expectEqualStrings("u8", typecheck.pattern_types[a_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[b_pattern.index()].kind());
    try testing.expectEqualStrings("u8", typecheck.pattern_types[b_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[c_pattern.index()].kind());
    try testing.expectEqualStrings("u8", typecheck.pattern_types[c_pattern.index()].name().?);
}

test "compiler pins integer coercion and error matrix" {
    const source_text =
        \\pub fn declaredOverflow() {
        \\    let x: u8 = 300;
        \\    _ = x;
        \\}
        \\
        \\pub fn contextOverflow() -> u8 {
        \\    return 256;
        \\}
        \\
        \\pub fn negativeToUnsigned() -> u8 {
        \\    return -1;
        \\}
        \\
        \\pub fn mixedArithmetic(a: u32, b: u64) -> u64 {
        \\    return a + b;
        \\}
        \\
        \\pub fn mixedComparison(a: u32, b: i32) -> bool {
        \\    return a < b;
        \\}
        \\
        \\pub fn mixedBitwise(a: u32, b: u64) -> u32 {
        \\    return a & b;
        \\}
        \\
        \\pub fn peerOverflow(x: u8) -> u8 {
        \\    return 1231231231 + x;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "constant value 300 does not fit in type 'u8'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "constant value 256 does not fit in type 'u8'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "constant value -1 does not fit in type 'u8'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "invalid binary operator '+' for types 'u32' and 'u64'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "invalid binary operator '<' for types 'u32' and 'i32'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "invalid binary operator '&' for types 'u32' and 'u64'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "constant value 1231231231 does not fit in type 'u8'"));
}

test "compiler rejects directly recursive runtime structs" {
    const source_text =
        \\struct Node {
        \\    next: Node,
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
}

test "compiler rejects mutually recursive runtime structs" {
    const source_text =
        \\struct A {
        \\    b: B,
        \\}
        \\
        \\struct B {
        \\    a: A,
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
}

test "compiler rejects product-contained recursive runtime structs" {
    const source_text =
        \\struct Node {
        \\    tuple_path: (u256, Node),
        \\    array_path: [Node; 2],
        \\    anonymous_path: struct { child: Node },
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
}

test "compiler rejects invalid constructor shape" {
    const source_text =
        \\contract Entry {
        \\    pub fn init(self, owner: address) -> bool {
        \\        return true;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!module_typecheck.diagnostics.isEmpty());
}

test "compiler rejects public ABI signatures using payload-carrying enums" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\contract Entry {
        \\    pub fn accept(value: Event) -> u256 {
        \\        return 0;
        \\    }
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "unsupported ABI type");
}

test "compiler package driver surfaces payload enum ABI diagnostics before HIR lowering" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\contract Entry {
        \\    pub fn accept(value: Event) -> u256 {
        \\        return 0;
        \\    }
        \\
        \\    pub fn make(flag: bool) -> Event {
        \\        if (flag) return Event.Value(1);
        \\        return Event.Empty;
        \\    }
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "public function parameter 'value' uses unsupported ABI type 'Event'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "public function 'make' uses unsupported return ABI type 'Event'"));
}

test "compiler rejects payload enum constructors with wrong arity" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Pair(u256, u256),
        \\}
        \\
        \\fn bad() -> Event {
        \\    return Event.Pair(1);
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "expected 2 arguments, found 1");
}

test "compiler rejects payload enum constructors with wrong payload type" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\fn bad(flag: bool) -> Event {
        \\    return Event.Value(flag);
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "expected type 'u256', found 'bool'");
}

test "compiler rejects payload enum match patterns with wrong destructure arity" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Pair(u256, u256),
        \\}
        \\
        \\fn bad(value: Event) -> u256 {
        \\    return switch (value) {
        \\        Event.Empty => 0,
        \\        Event.Pair(one) => one,
        \\    };
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "ADT tuple payload bindings must match the payload field count");
}

test "compiler rejects named payload enum field literals with missing unknown duplicate or wrong typed fields" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Named { code: u256, amount: u256 },
        \\}
        \\
        \\fn missing(code: u256) -> Event {
        \\    return Event.Named { code: code };
        \\}
        \\
        \\fn unknown(code: u256, amount: u256) -> Event {
        \\    return Event.Named { code: code, amount: amount, extra: amount };
        \\}
        \\
        \\fn duplicate(code: u256, amount: u256) -> Event {
        \\    return Event.Named { code: code, amount: amount, code: amount };
        \\}
        \\
        \\fn wrong_type(flag: bool, amount: u256) -> Event {
        \\    return Event.Named { code: flag, amount: amount };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "missing field 'amount' for ADT variant 'Named'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "unknown field 'extra' for ADT variant 'Named'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "duplicate field 'code' for ADT variant 'Named'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "expected type 'u256', found 'bool'"));
}

test "compiler rejects named payload enum structural match patterns with missing unknown or duplicate fields" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Named { code: u256, amount: u256 },
        \\}
        \\
        \\fn missing(value: Event) -> u256 {
        \\    return switch (value) {
        \\        Event.Empty => 0,
        \\        Event.Named { code } => code,
        \\    };
        \\}
        \\
        \\fn unknown(value: Event) -> u256 {
        \\    return switch (value) {
        \\        Event.Empty => 0,
        \\        Event.Named { code, extra } => code,
        \\    };
        \\}
        \\
        \\fn duplicate(value: Event) -> u256 {
        \\    return switch (value) {
        \\        Event.Empty => 0,
        \\        Event.Named { code, code: other } => code,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(ast_diags, "duplicate destructuring field name 'code'"));

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "ADT named payload destructure must bind every payload field"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "missing ADT payload field 'amount'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "unknown ADT payload field 'extra'"));
}

test "compiler warns when wildcard enum match arm covers named variants" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\fn classify(value: Event) -> u256 {
        \\    return switch (value) {
        \\        Event.Value(_) => 1,
        \\        _ => 0,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "wildcard match arm covers named variants"));
}

test "compiler rejects enum or-pattern alternatives with mismatched bindings" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\fn classify(event: Event) -> u256 {
        \\    return switch (event) {
        \\        Event.Value(x) | Event.Empty => x,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "or-pattern alternatives must bind the same names"));
}

test "compiler rejects enum or-pattern alternatives with incompatible binding types" {
    const source_text =
        \\enum Event {
        \\    Number(u256),
        \\    Flag(bool),
        \\}
        \\
        \\fn classify(event: Event) -> u256 {
        \\    return switch (event) {
        \\        Event.Number(x) | Event.Flag(x) => 1,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "or-pattern binding 'x' has incompatible types"));
}

test "compiler rejects directly recursive payload-carrying enums" {
    const source_text =
        \\enum Tree {
        \\    Leaf,
        \\    Node(Tree),
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
}
