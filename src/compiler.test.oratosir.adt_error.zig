const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const compileText = common.compileText;
const renderSirTextForModule = common.renderSirTextForModule;
const expectNoResidualOraRuntimeOps = common.expectNoResidualOraRuntimeOps;
const functionSlice = common.functionSlice;
const expectOrderedNeedles = common.expectOrderedNeedles;

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
    // mload256 v1 (lhs/code) plus mload256 (v1+0x20) (rhs/amount). With CSE
    // enabled, the 0x20 offset can be shared across both variants.
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "const 0x20"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "= add v1 "));
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

test "compiler converts ADT storage load and store through carrier slots" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\    Pair(u256, u256),
        \\}
        \\
        \\contract Entry {
        \\    storage var saved: Event;
        \\
        \\    pub fn set(amount: u256) {
        \\        saved = Event.Value(amount);
        \\    }
        \\
        \\    pub fn read() -> u256 {
        \\        return match (saved) {
        \\            Event.Empty => 0,
        \\            Event.Value(x) => x,
        \\            Event.Pair(a, _) => a,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "sstore"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "sload"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts scalar Result storage load and store through carrier slots" {
    const source_text =
        \\error Failure;
        \\
        \\contract ResultStorage {
        \\    storage var saved: Result<u256, Failure>;
        \\
        \\    pub fn set_ok(value: u256) {
        \\        saved = Ok(value);
        \\    }
        \\
        \\    pub fn set_err() {
        \\        saved = Err(Failure());
        \\    }
        \\
        \\    pub fn get() -> u256 {
        \\        return match (saved) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "sstore"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "sload"));
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
    const read_pair = try functionSlice(rendered, "read_pair");
    try expectOrderedNeedles(read_pair, &.{ "mload256 v0", "mload256", "add", "mload256" });
    try testing.expect(std.mem.containsAtLeast(u8, read_pair, 1, "const 0x20"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "= and v1 "));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "= shr v1 "));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "OraToSIR lowers struct_field_store with declaration field index layout" {
    const ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(ctx);
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
    _ = mlir.oraDialectRegister(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    "ora.struct.decl"() ({
        \\    }) {name = "Pair", sym_name = "Pair", ora.field_names = ["first", "second"], ora.field_types = [!ora.int<256, false>, !ora.int<256, false>]} : () -> ()
        \\    func.func @store(%pair: !ora.struct<"Pair">, %value: !ora.int<256, false>) {
        \\      "ora.struct_field_store"(%pair, %value) {field_name = "second"} : (!ora.struct<"Pair">, !ora.int<256, false>) -> ()
        \\      ora.return
        \\    }
        \\  }
        \\}
    ;
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    defer mlir.oraModuleDestroy(module);
    try testing.expect(!mlir.oraModuleIsNull(module));
    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.addptr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 32 : !sir.u256"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.struct_field_store"));
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

test "compiler converts wide explicit enum constants through OraToSIR" {
    const source_text =
        \\enum Big : u256 {
        \\    A = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
        \\}
        \\
        \\fn current() -> Big {
        \\    return Big.A;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(
        std.mem.containsAtLeast(u8, rendered, 1, "large_const 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF") or
            std.mem.containsAtLeast(u8, rendered, 1, "sir.const 115792089237316195423570985008687907853269984665640564039457584007913129639935"),
    );
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.const -1"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.const 0"));
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

test "compiler stores opaque Result values into local Result memrefs through OraToSIR" {
    const source_text =
        \\error Failure;
        \\
        \\contract ResultMemRefOpaque {
        \\    fn choose(flag: bool, value: u256) -> Result<u256, Failure> {
        \\        if (flag) {
        \\            return Ok(value);
        \\        }
        \\        return Err(Failure());
        \\    }
        \\
        \\    pub fn run(flag: bool, value: u256) -> u256 {
        \\        var values: [Result<u256, Failure>; 2] = [Ok(0), Err(Failure())];
        \\        values[0] = choose(flag, value);
        \\        return match (values[0]) {
        \\            Ok(inner) => inner,
        \\            Err(_) => 99,
        \\        };
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn choose:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn run:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "icall @choose"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "alloc_size = const 0x80"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mstore256 elem4_ptr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mstore256 elem5_ptr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "elem9 = mload256"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "shl c1 v"));
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

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn std_bytes_at:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn std_bytes_decodeU256BE:"));
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

test "compiler limits public error-union dispatcher to declared return errors" {
    const source_text =
        \\error OnlyA();
        \\error OnlyB();
        \\error Payload(code: u256);
        \\
        \\contract Probe {
        \\    pub fn one(flag: bool) -> !bool | OnlyA {
        \\        if (flag) {
        \\            return true;
        \\        }
        \\        return error OnlyA();
        \\    }
        \\
        \\    pub fn two(flag: bool) -> !bool | OnlyB | Payload {
        \\        if (flag) {
        \\            return error OnlyB();
        \\        }
        \\        return error Payload(7);
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

    const main_fn = try functionSlice(rendered, "main");
    const only_a = try std.fmt.allocPrint(testing.allocator, "0x{X:0>8}", .{compiler.hir.abi.keccakSelectorValue("OnlyA()")});
    defer testing.allocator.free(only_a);
    const only_b = try std.fmt.allocPrint(testing.allocator, "0x{X:0>8}", .{compiler.hir.abi.keccakSelectorValue("OnlyB()")});
    defer testing.allocator.free(only_b);
    const payload = try std.fmt.allocPrint(testing.allocator, "0x{X:0>8}", .{compiler.hir.abi.keccakSelectorValue("Payload(uint256)")});
    defer testing.allocator.free(payload);

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, main_fn, only_a));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, main_fn, only_b));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, main_fn, payload));
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

test "compiler lowers message assert to selector revert payload through OraToSIR" {
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
    const selector = try compiler.hir.abi.keccakSelectorHex(testing.allocator, "bad");
    defer testing.allocator.free(selector);

    const hir_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (hir_text_ref.data != null) mlir.oraStringRefFree(hir_text_ref);
    const hir_rendered = hir_text_ref.data[0..hir_text_ref.length];
    try testing.expect(std.mem.containsAtLeast(u8, hir_rendered, 1, "ora.assert_selector"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_rendered, 1, selector));

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "const 4"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.store8"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "08C379A0"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "08c379a0"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.assert"));
}
