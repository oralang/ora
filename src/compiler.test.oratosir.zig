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

test "compiler lowers Err discard patterns through OraToSIR" {
    const source_text =
        \\error Failure(code: u256);
        \\error Denied(owner: address);
        \\
        \\contract Probe {
        \\    fn decide(flag: bool) -> !u256 | Failure | Denied {
        \\        if (flag) {
        \\            return 10;
        \\        }
        \\        return Failure(7);
        \\    }
        \\
        \\    pub fn run(flag: bool) -> u256 {
        \\        return match (decide(flag)) {
        \\            Ok(_) => 1,
        \\            Err(_) => 2,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn run:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "c2 = const 0x2"));
}

test "compiler lowers guard clauses to runtime revert through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(amount: u256) -> bool
        \\        guard amount < 10;
        \\    {
        \\        return true;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.invalid"));
}

test "compiler converts tuple top-level const items through OraToSIR" {
    const source_text =
        \\const PAIR: (u256, u256) = @divmod(17, 5);
        \\
        \\fn run() -> u256 {
        \\    return PAIR.0 * 5 + PAIR.1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const sir_text = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(sir_text);
    try expectNoResidualOraRuntimeOps(sir_text);
}

test "compiler converts aggregate ADT payload matches through compiler-managed payload handles" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Pair(u256, u256),
        \\    Named { code: u256, amount: u256 },
        \\}
        \\
        \\fn sum(event: Event) -> u256 {
        \\    return switch (event) {
        \\        Event.Empty => 0,
        \\        Event.Pair(lhs, rhs) => lhs + rhs,
        \\        Event.Named(code, amount) => code + amount,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    // Function takes the wide ADT carrier as two u256 args (v0=tag, v1=payload handle).
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn sum:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "entry v0 v1"));
    // Both Pair and Named payloads are 2-tuples dereferenced through the handle:
    // mload256 v1 (lhs/code) plus mload256 (v1+0x20) (rhs/amount). With both
    // variants taking that path, expect at least four mload256 reads and at
    // least two const 0x20 offsets.
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "const 0x20"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts named aggregate ADT payload structural matches through OraToSIR" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Named { code: u256, amount: u256 },
        \\}
        \\
        \\fn make() -> Event {
        \\    return Event.Named {
        \\        amount: 20,
        \\        code: 10,
        \\    };
        \\}
        \\
        \\fn sum(event: Event) -> u256 {
        \\    return switch (event) {
        \\        Event.Empty => 0,
        \\        Event.Named { amount: value, code } => value + code,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn sum:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "mload256"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts nested ADT fields through compiler-managed handle storage" {
    const source_text =
        \\struct Holder {
        \\    event: Event,
        \\}
        \\
        \\enum Event {
        \\    Empty,
        \\    Pair(u256, u256),
        \\    Named { code: u256, amount: u256 },
        \\}
        \\
        \\fn read_pair(holder: Holder) -> u256 {
        \\    return switch (holder.event) {
        \\        Event.Empty => 0,
        \\        Event.Pair(lhs, rhs) => lhs + rhs,
        \\        Event.Named(code, amount) => code + amount,
        \\    };
        \\}
        \\
        \\fn forward(event: Event) -> Event {
        \\    return event;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    // The struct stores a handle pointer to a 64-byte (tag, payload) carrier.
    // Reading holder.event must dereference the handle:
    //   v1 = mload256 v0           ; load handle pointer from struct field
    //   v2 = mload256 v1           ; load tag word at offset 0
    //   ... = mload256 (v1 + 0x20) ; load payload word at offset 32
    // NEVER apply a narrow `& 1` / `>> 1` decode of the loaded field —
    // Event has 3 variants and aggregate payloads, so narrow packing is
    // invalid (the Named arm becomes unreachable).
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn read_pair:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "v1 = mload256 v0"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "v2 = mload256 v1"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "= and v1 "));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "= shr v1 "));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts source scalar ADT constructors through OraToSIR" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\    Pair(u256, u256),
        \\}
        \\
        \\fn choose(flag: bool) -> Event {
        \\    if (flag) {
        \\        return Event.Value(7);
        \\    }
        \\    return Event.Pair(2, 3);
        \\}
        \\
        \\fn classify(flag: bool) -> u256 {
        \\    return switch (choose(flag)) {
        \\        Event.Empty => 0,
        \\        Event.Value(value) => value,
        \\        Event.Pair(lhs, rhs) => lhs + rhs,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn choose:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn classify:"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts source aggregate ADT constructors through OraToSIR" {
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
        \\fn choose(flag: bool) -> Event {
        \\    if (flag) {
        \\        return Event.Wrapped(Receipt { code: 10, amount: 20 });
        \\    }
        \\    return Event.Named(5, 6);
        \\}
        \\
        \\fn project(flag: bool) -> u256 {
        \\    return switch (choose(flag)) {
        \\        Event.Empty => 0,
        \\        Event.Wrapped(receipt) => receipt.code,
        \\        Event.Named(code, amount) => amount,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn choose:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn project:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "mload256"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts native string and bytes len field access through OraToSIR" {
    const source_text =
        \\pub fn string_len(text: string) -> u256 {
        \\    return text.len;
        \\}
        \\
        \\pub fn bytes_len(data: bytes) -> u256 {
        \\    return data.len;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "mload256"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts native string and bytes index access through OraToSIR" {
    const source_text =
        \\pub fn string_at(text: string, i: u256) -> u8 {
        \\    return text[i];
        \\}
        \\
        \\pub fn bytes_at(data: bytes, i: u256) -> u8 {
        \\    return data[i];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "mload8"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "0xFF"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler resolves imported std bytes errors in type position and lowers them through OraToSIR" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Probe {
        \\    pub fn first(data: bytes) -> !u8 | std.bytes.OutOfBounds {
        \\        return std.bytes.at(data, 0);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn std_bytes_at:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload8"));
}

test "compiler preserves error selectors through OraToSIR" {
    const source_text =
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract Vault {
        \\    pub fn withdraw(amount: u256) -> !u256 | InsufficientBalance {
        \\        return error InsufficientBalance(amount, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "cf479181"));
}

test "compiler lowers payload error return constructors through OraToSIR" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn run() -> !bool | Failure {
        \\        return error Failure(7);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.return"));
}

test "compiler lowers bare assert to runtime revert through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(flag: bool) {
        \\        assert(flag);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.invalid"));
}

test "compiler lowers message assert to runtime revert payload through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(flag: bool) {
        \\        assert(flag, "bad");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.store8"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.assert"));
}

test "compiler examples convert through SIR" {
    const example_paths = [_][]const u8{
        "ora-example/smoke.ora",
        "ora-example/no_return_test.ora",
        "ora-example/dce_test.ora",
        "ora-example/statements/contract_declaration.ora",
    };

    for (example_paths) |path| {
        try expectOraToSirConverts(path);
    }
}

test "compiler converts contract storage through explicit slot metadata" {
    const source_text =
        \\contract Vault {
        \\    storage var balance: u256 = 1;
        \\    storage var owner: address;
        \\
        \\    pub fn read() -> u256 {
        \\        return balance;
        \\    }
        \\
        \\    pub fn write(next: u256, who: address) {
        \\        balance = next;
        \\        owner = who;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "slot_balance"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "slot_owner"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.sload"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.sstore"));
}

test "compiler converts narrowed carried locals in nested scf ifs" {
    const source_text =
        \\contract Test {
        \\    pub fn update(current_status: u8, user_borrow: u256, health: u256) -> u8 {
        \\        var new_status: u8 = current_status;
        \\        if (current_status == 0) {
        \\            if (user_borrow == 0) {
        \\                new_status = 1;
        \\            } else {
        \\                if (health < 10000) {
        \\                    new_status = 3;
        \\                }
        \\            }
        \\        }
        \\        return new_status;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
}

test "compiler examples leave no residual Ora runtime ops after OraToSIR" {
    const example_paths = [_][]const u8{
        "ora-example/smoke.ora",
        "ora-example/no_return_test.ora",
        "ora-example/dce_test.ora",
        "ora-example/statements/contract_declaration.ora",
        "ora-example/apps/erc20.ora",
        "ora-example/apps/counter.ora",
        "ora-example/apps/arithmetic_probe.ora",
        "ora-example/apps/erc20_verified.ora",
        "ora-example/apps/defi_lending_pool.ora",
        "ora-example/apps/erc20_bitfield_comptime_generics.ora",
        "ora-example/array_operations.ora",
        "ora-example/structs/basic_structs.ora",
        "ora-example/tuples/tuple_basics.ora",
        "ora-example/bitfields/basic_bitfield_storage.ora",
        "ora-example/bitfields/bitfield_map_values.ora",
        "ora-example/comptime/comptime_basics.ora",
        "ora-example/comptime/comptime_functions.ora",
        "ora-example/errors/try_catch.ora",
        "ora-example/locks/lock_runtime_map_guard.ora",
        "ora-example/locks/lock_runtime_scalar_guard.ora",
        "ora-example/locks/lock_runtime_independent_roots.ora",
        "ora-example/regions/region_ok_storage_map.ora",
        "ora-example/regions/region_ok_storage_tstore_map_and_scalar_writes.ora",
        "ora-example/regions/region_ok_storage_tstore_same_type_writes.ora",
        "ora-example/refinements/dispatcher_refinement_e2e.ora",
        "ora-example/refinements/basic_refinements.ora",
        "ora-example/refinements/comprehensive_test.ora",
        "ora-example/refinements/guards_showcase.ora",
        "ora-example/smt/soundness/conditional_return_split.ora",
        "ora-example/smt/soundness/fail_loop_invariant_post.ora",
        "ora-example/smt/soundness/overflow_mul_constant.ora",
        "ora-example/smt/soundness/switch_arm_path_predicates.ora",
        "ora-example/smt/verification/state_invariants.ora",
        "ora-example/vault/02_errors.ora",
    };

    for (example_paths) |path| {
        var compilation = try compilePackage(path);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try expectNoResidualOraRuntimeOps(rendered);
    }
}
