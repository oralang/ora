const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const runtime_checks = common.runtime_checks;
const compileText = common.compileText;
const renderOraMlirForSource = common.renderOraMlirForSource;
const renderSirTextForModule = common.renderSirTextForModule;
const expectOraToSirConverts = common.expectOraToSirConverts;
const extractSirGlobalSlotsJson = common.extractSirGlobalSlotsJson;
const expectGlobalSlot = common.expectGlobalSlot;
const createOraMlirContext = common.createOraMlirContext;
const parseOraModule = common.parseOraModule;
const functionSlice = common.functionSlice;
const oraFunctionSlice = common.oraFunctionSlice;
const expectOrderedNeedles = common.expectOrderedNeedles;

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

test "compiler permits ambiguous duplicate storage names in syntax examples" {
    const source_text =
        \\contract First {
        \\    storage var owner: address;
        \\}
        \\
        \\contract Second {
        \\    storage var owner: address;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.global_slot_ambiguous_names"));
}

test "OraToSIR rejects mismatched strict storage slot metadata" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module attributes {ora.global_slots_built, ora.global_slots = {counter = 0 : ui64}} {
        \\  ora.global "counter" : !ora.int<256, false> {ora.slot_index = 1 : ui64}
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(!mlir.oraConvertToSIR(ctx, module, false));
}

test "OraToSIR rejects missing strict storage slot metadata" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module attributes {ora.global_slots_built, ora.global_slots = {counter = 0 : ui64}} {
        \\  ora.global "counter" : !ora.int<256, false>
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(!mlir.oraConvertToSIR(ctx, module, false));
}

test "OraToSIR lowers named memory slots without release malloc fallback" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module attributes {ora.global_slots_built, ora.global_slots = {scratch = 0 : ui64}} {
        \\  ora.contract @C {
        \\    func.func @roundtrip(%arg0: !sir.u256) {
        \\      ora.mstore %arg0, "scratch" : !sir.u256
        \\      %0 = ora.mload "scratch" : !sir.u256
        \\      sir.iret %0
        \\    }
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));
    const rendered = try renderSirTextForModule(ctx, module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "codesize"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "malloc"));
}

test "SIR text legalizer rejects icall result underflow instead of zero filling" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @callee() {
        \\    sir.stop
        \\  }
        \\  func.func @main() {
        \\    %0 = sir.icall @callee() : !sir.u256
        \\    sir.stop
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(!mlir.oraLegalizeSIRText(ctx, module));
}

test "SIR dispatcher rejects icall result underflow instead of zero filling" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @callee() {
        \\    sir.stop
        \\  }
        \\  func.func @caller() {
        \\    %0 = sir.icall @callee() : !sir.u256
        \\    sir.stop
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(!mlir.oraBuildSIRDispatcher(ctx, module));
}

test "SIR text legalizer rejects ptr word icall result repair" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @callee() -> !sir.ptr<1> {
        \\    sir.stop
        \\  }
        \\  func.func @main() {
        \\    %0 = sir.icall @callee() : !sir.u256
        \\    sir.stop
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(!mlir.oraLegalizeSIRText(ctx, module));
}

test "SIR dispatcher rejects ptr word icall result repair" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @callee() -> !sir.ptr<1> {
        \\    sir.stop
        \\  }
        \\  func.func @caller() {
        \\    %0 = sir.icall @callee() : !sir.u256
        \\    sir.stop
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(!mlir.oraBuildSIRDispatcher(ctx, module));
}

test "OraToSIR rejects malformed ABI encode and decode layout attributes" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const malformed_layouts = [_][]const u8{
        "",
        "static(weird)",
        "tuple(",
        "tuple(static(uint256)",
        "tuple(static(uint256))extra",
        "array(abc,static(uint256))",
        "array(dynamic,dynamic(weird))",
    };

    const cases = [_]struct {
        fn_name: []const u8,
        body: []const u8,
    }{
        .{
            .fn_name = "encode",
            .body = "%encoded = \"ora.abi_encode\"(%value) {{layout = \"{s}\"}} : (!ora.int<256, false>) -> !ora.int<256, false>",
        },
        .{
            .fn_name = "decode",
            .body = "%decoded = \"ora.abi_decode\"(%returndata) {{return_types = [\"u256\"], layout = \"{s}\", source = \"returndata\", failure_mode = \"error_union\"}} : (!ora.bytes) -> !ora.int<256, false>",
        },
    };

    inline for (cases) |case| {
        for (malformed_layouts) |layout| {
            const body = try std.fmt.allocPrint(testing.allocator, case.body, .{layout});
            defer testing.allocator.free(body);
            const text = try std.fmt.allocPrint(testing.allocator,
                \\module {{
                \\  ora.contract @C {{
                \\    func.func @{s}(%value: !ora.int<256, false>, %returndata: !ora.bytes) {{
                \\      {s}
                \\      ora.return
                \\    }}
                \\  }}
                \\}}
            , .{ case.fn_name, body });
            defer testing.allocator.free(text);

            const module = try parseOraModule(ctx, text);
            defer mlir.oraModuleDestroy(module);

            try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
            try testing.expect(!mlir.oraConvertToSIR(ctx, module, false));
        }
    }
}

test "OraToSIR accepts bare single-value ABI layout attributes" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @encode(%value: !ora.int<256, false>) {
        \\      %encoded = "ora.abi_encode"(%value) {layout = "static(uint256)"} : (!ora.int<256, false>) -> !ora.int<256, false>
        \\      ora.return
        \\    }
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.malloc"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_encode"));
}

test "compiler storage layout manifest matches SIR slot usage" {
    const source_text =
        \\contract LayoutProbe {
        \\    storage var balance: u256 = 0;
        \\    storage var owner: address;
        \\    storage var balances: map<address, u256>;
        \\    storage var allowances: map<address, map<address, u256>>;
        \\    storage var history: [u256; 4];
        \\    storage var after_history: u256;
        \\
        \\    pub fn write(who: address, spender: address, index: u256, amount: u256) {
        \\        balance = amount;
        \\        balances[who] = amount;
        \\        allowances[who][spender] = amount;
        \\        history[index] = amount;
        \\        after_history = amount;
        \\    }
        \\
        \\    pub fn read(who: address, index: u256) -> u256 {
        \\        return balance + balances[who] + history[index] + after_history;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const slots_json = try extractSirGlobalSlotsJson(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(slots_json);
    try expectGlobalSlot(slots_json, "balance", 0);
    try expectGlobalSlot(slots_json, "owner", 1);
    try expectGlobalSlot(slots_json, "balances", 2);
    try expectGlobalSlot(slots_json, "allowances", 3);
    try expectGlobalSlot(slots_json, "history", 4);
    try expectGlobalSlot(slots_json, "after_history", 8);

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const write_fn = try functionSlice(rendered, "write");
    try expectOrderedNeedles(write_fn, &.{
        "slot_balances",
        "mstore256 ptr",
        "mstore256 ptr_off slot_balances",
        "keccak256",
        "sstore",
    });
    try expectOrderedNeedles(write_fn, &.{
        "slot_allowances",
        "mstore256 ptr_",
        "mstore256 ptr_off_",
        "slot_allowances",
        "keccak256",
        "mstore256 ptr_",
        "mstore256 ptr_off_",
        "keccak256",
        "sstore",
    });
    try expectOrderedNeedles(write_fn, &.{
        "slot_history",
        "mul",
        "add slot_history",
        "sstore",
    });

    const read_fn = try functionSlice(rendered, "read");
    try expectOrderedNeedles(read_fn, &.{
        "slot_history",
        "mul",
        "add 0x4",
        "sload",
    });
}

test "compiler does not store nested map handles back to parent slots" {
    const source_text =
        \\contract NestedMapProbe {
        \\    storage var allowances: map<address, map<address, u256>>;
        \\
        \\    pub fn direct(owner: address, spender: address, amount: u256) {
        \\        allowances[owner][spender] = amount;
        \\    }
        \\}
    ;

    const ora_text = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(ora_text);

    const direct_ora = try oraFunctionSlice(ora_text, "direct");
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, direct_ora, "ora.map_store"));

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const direct_sir = try functionSlice(rendered, "direct");
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, direct_sir, "sstore"));
    try expectOrderedNeedles(direct_sir, &.{
        "slot_allowances",
        "keccak256",
        "keccak256",
        "sstore",
    });
}

test "OraToSIR lowers direct transient resource places to tload and tstore" {
    const source_text =
        \\resource TokenUnit = u256;
        \\comptime const std = @import("std");
        \\
        \\contract TransientResourceProbe {
        \\    tstore var scratch: Resource<TokenUnit>;
        \\
        \\    pub fn issue(amount: TokenUnit)
        \\        requires @amount(scratch) <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\    {
        \\        @create(scratch, amount);
        \\    }
        \\
        \\    pub fn retire(amount: TokenUnit)
        \\        requires @amount(scratch) >= amount
        \\    {
        \\        @destroy(scratch, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const issue = try functionSlice(rendered, "issue");
    try testing.expect(std.mem.containsAtLeast(u8, issue, 1, "tload"));
    try testing.expect(std.mem.containsAtLeast(u8, issue, 1, "tstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, issue, 1, "sload"));
    try testing.expect(!std.mem.containsAtLeast(u8, issue, 1, "sstore"));

    const retire = try functionSlice(rendered, "retire");
    try testing.expect(std.mem.containsAtLeast(u8, retire, 1, "tload"));
    try testing.expect(std.mem.containsAtLeast(u8, retire, 1, "tstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, retire, 1, "sload"));
    try testing.expect(!std.mem.containsAtLeast(u8, retire, 1, "sstore"));
}

test "OraToSIR lowers signed resource guards with signed comparisons" {
    const source_text =
        \\resource DebtUnit = i256;
        \\
        \\contract SignedResourceProbe {
        \\    storage var debts: map<address, Resource<DebtUnit>>;
        \\    storage var settled: Resource<DebtUnit>;
        \\
        \\    pub fn mint(owner: address, amount: DebtUnit)
        \\        modifies debts[owner]
        \\        requires amount >= 0
        \\        requires amount <= 100
        \\        requires @amount(debts[owner]) <= 100
        \\    {
        \\        @create(debts[owner], amount);
        \\    }
        \\
        \\    pub fn settle(from: address, amount: DebtUnit)
        \\        modifies debts[from], settled
        \\        requires amount >= 0
        \\        requires amount <= 100
        \\        requires @amount(debts[from]) >= -100
        \\        requires @amount(settled) <= 100
        \\    {
        \\        @move(debts[from], settled, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const mint = try functionSlice(rendered, "mint");
    try testing.expect(std.mem.containsAtLeast(u8, mint, 1, "slt"));

    const settle = try functionSlice(rendered, "settle");
    try testing.expect(std.mem.containsAtLeast(u8, settle, 1, "slt"));
    try testing.expect(std.mem.containsAtLeast(u8, settle, 1, "sgt"));
}

test "OraToSIR skips resource fallback guards after verified marker" {
    const source_text =
        \\resource TokenUnit = u256;
        \\
        \\contract ResourceGuardDedup {
        \\    storage var left: Resource<TokenUnit>;
        \\    storage var right: Resource<TokenUnit>;
        \\
        \\    pub fn transfer(amount: TokenUnit)
        \\        modifies left, right
        \\    {
        \\        @move(left, right, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    runtime_checks.markVerifiedResourceRuntimeChecks(hir_result.context, hir_result.module.raw_module);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const transfer = try functionSlice(rendered, "transfer");
    try testing.expect(std.mem.containsAtLeast(u8, transfer, 2, "sload"));
    try testing.expect(std.mem.containsAtLeast(u8, transfer, 1, "sub"));
    try testing.expect(std.mem.containsAtLeast(u8, transfer, 1, "add"));
    try testing.expect(std.mem.containsAtLeast(u8, transfer, 2, "sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, transfer, 1, "lt"));
    try testing.expect(!std.mem.containsAtLeast(u8, transfer, 1, "iszero"));
}

test "OraToSIR reuses resource map place hash for self move" {
    const source_text =
        \\resource TokenUnit = u256;
        \\
        \\contract ResourcePlaceCse {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn self_move(owner: address, amount: TokenUnit)
        \\        modifies balances[owner]
        \\        requires @amount(balances[owner]) >= amount
        \\    {
        \\        @move(balances[owner], balances[owner], amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const self_move = try functionSlice(rendered, "self_move");
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, self_move, "keccak256"));
    try testing.expect(!std.mem.containsAtLeast(u8, self_move, 1, "sstore"));
}

test "OraToSIR lowers lock and guard to matching transient key shapes" {
    const source_text =
        \\contract GuardedWrites {
        \\    storage var total: u256;
        \\    storage var history: [u256; 8];
        \\
        \\    fn write_total(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    fn write_history(index: u256, value: u256) {
        \\        history[index] = value;
        \\    }
        \\
        \\    pub fn touch_total(value: u256) {
        \\        @lock(total);
        \\        write_total(value);
        \\    }
        \\
        \\    pub fn touch_history(index: u256, value: u256) {
        \\        @lock(history[index]);
        \\        write_history(index, value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const lock_prefix = "large_const 0x8000000000000000000000000000000000000000000000000000000000000000";

    const touch_total = try functionSlice(rendered, "touch_total");
    try expectOrderedNeedles(touch_total, &.{
        lock_prefix,
        "tload",
        "revert",
        "tstore",
    });

    const write_total = try functionSlice(rendered, "write_total");
    try expectOrderedNeedles(write_total, &.{
        "slot_total = const 0x0",
        lock_prefix,
        "tload",
        "revert",
        "sstore 0x0",
    });

    const touch_history = try functionSlice(rendered, "touch_history");
    try expectOrderedNeedles(touch_history, &.{
        "mstore256",
        "keccak256",
        "add",
        "tload",
        "revert",
        "tstore",
    });
    try testing.expect(std.mem.containsAtLeast(u8, touch_history, 1, lock_prefix));

    const write_history = try functionSlice(rendered, "write_history");
    try expectOrderedNeedles(write_history, &.{
        "slot_history = const 0x1",
        "mstore256",
        "keccak256",
        "add",
        "tload",
        "revert",
        "sstore",
    });
    try testing.expect(std.mem.containsAtLeast(u8, write_history, 1, lock_prefix));

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "tstore"));
    try testing.expectEqual(@as(usize, 4), std.mem.count(u8, rendered, "tload"));
    try testing.expectEqual(@as(usize, 4), std.mem.count(u8, rendered, lock_prefix));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "keccak256"));
}

test "OraToSIR lowers deep dynamic struct scalar update to direct field sstore" {
    const source_text =
        \\struct Leaf {
        \\    left: u256;
        \\    values: slice[u256];
        \\    right: u256;
        \\}
        \\
        \\struct Mid {
        \\    before: u256;
        \\    leaf: Leaf;
        \\    after: u256;
        \\}
        \\
        \\struct Outer {
        \\    head: u256;
        \\    mid: Mid;
        \\    tail: u256;
        \\}
        \\
        \\contract DeepDynamicStructStorageSmoke {
        \\    storage var record: Outer;
        \\
        \\    pub fn set_leaf_right(value: u256) {
        \\        record.mid.leaf.right = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const update_fn = try functionSlice(rendered, "set_leaf_right");
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, update_fn, "sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, update_fn, 1, "keccak256"));
}

test "OraToSIR lowers dynamic string and bytes storage map values through storage roots" {
    const source_text =
        \\contract DynamicMapValues {
        \\    storage var names: map<address, string>;
        \\    storage var blobs: map<u256, bytes>;
        \\
        \\    pub fn setName(account: address, name: string) {
        \\        names[account] = name;
        \\    }
        \\
        \\    pub fn getName(account: address) -> string {
        \\        return names[account];
        \\    }
        \\
        \\    pub fn copyBlob(from: u256, to: u256) {
        \\        blobs[to] = blobs[from];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const set_name = try functionSlice(rendered, "setName");
    try testing.expect(std.mem.count(u8, set_name, "sstore") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, set_name, 1, "mload"));

    const get_name = try functionSlice(rendered, "getName");
    try testing.expect(std.mem.count(u8, get_name, "sload") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, get_name, 1, "malloc"));

    const copy_blob = try functionSlice(rendered, "copyBlob");
    try testing.expect(std.mem.count(u8, copy_blob, "keccak256") >= 2);
    try testing.expect(std.mem.count(u8, copy_blob, "sload") >= 2);
    try testing.expect(std.mem.count(u8, copy_blob, "sstore") >= 2);
}

test "OraToSIR lowers dynamic string storage map keys" {
    const source_text =
        \\contract MapStringKey {
        \\    storage var values: map<string, u256>;
        \\
        \\    pub fn set(key: string, val: u256) {
        \\        values[key] = val;
        \\    }
        \\
        \\    pub fn get(key: string) -> u256 {
        \\        return values[key];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const set_fn = try functionSlice(rendered, "set");
    try testing.expect(std.mem.count(u8, set_fn, "keccak256") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, set_fn, 1, "sstore"));

    const get_fn = try functionSlice(rendered, "get");
    try testing.expect(std.mem.count(u8, get_fn, "keccak256") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, get_fn, 1, "sload"));
}

test "OraToSIR lowers fixed array storage map values" {
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
        \\
        \\    pub fn set_all(account: address, a: u256, b: u256, c: u256, d: u256) {
        \\        let arr: [u256; 4] = [a, b, c, d];
        \\        slots[account] = arr;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const get_slot = try functionSlice(rendered, "get_slot");
    try testing.expect(std.mem.containsAtLeast(u8, get_slot, 1, "sload"));

    const set_all = try functionSlice(rendered, "set_all");
    try testing.expect(std.mem.count(u8, set_all, "sstore") >= 4);
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

test "OraToSIR lowers bytes Result unwrap_or helper call" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\error Failure();
        \\
        \\contract ResultHelpers {
        \\    fn choose_bytes(flag: bool, value: bytes) -> Result<bytes, Failure> {
        \\        if (flag) {
        \\            return Ok(value);
        \\        }
        \\        return Err(Failure());
        \\    }
        \\
        \\    pub fn bytes_or_self(flag: bool, value: bytes) -> bytes {
        \\        let maybe = choose_bytes(flag, value);
        \\        return std.result.unwrap_or(maybe, value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
}
