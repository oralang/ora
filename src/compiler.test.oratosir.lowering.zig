const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const compileText = common.compileText;
const renderOraMlirForSource = common.renderOraMlirForSource;
const renderSirTextForModule = common.renderSirTextForModule;
const functionSlice = common.functionSlice;
const expectOrderedNeedles = common.expectOrderedNeedles;

test "compiler lowers indexed event fields as EVM log topics" {
    const source_text =
        \\contract Logs {
        \\    log Transfer(indexed from: address, indexed to: address, amount: u256);
        \\
        \\    pub fn run(from: address, to: address, amount: u256) {
        \\        log Transfer(from, to, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(std.mem.containsAtLeast(u8, run_fn, 1, "log3"));
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "log1"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, run_fn, "mstore256"));
    try expectOrderedNeedles(run_fn, &.{ "const 0x20", "malloc", "mstore256", "log3" });
}

test "compiler lowers runtime keccak256 through OraToSIR" {
    const source_text =
        \\pub fn hash(data: bytes) -> u256 {
        \\    return @keccak256(data);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "keccak256"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.keccak256"));
}

test "compiler lowers computed storage derive and word operations through OraToSIR" {
    const source_text =
        \\contract Vault {
        \\    pub fn write(owner: address, offset: u256, value: u256) {
        \\        let slot: StorageSlot = @storageDerive("vault.position", owner);
        \\        @storageWordStore(slot, offset, value);
        \\    }
        \\
        \\    pub fn read(owner: address, offset: u256) -> u256 {
        \\        let slot: StorageSlot = @storageDerive("vault.position", owner);
        \\        return @storageWordLoad(slot, offset);
        \\    }
        \\
        \\    pub fn clear(owner: address) {
        \\        let slot: StorageSlot = @storageDerive("vault.position", owner);
        \\        let range: StorageRange = @storageRange(slot, 2);
        \\        @storageRangeErase(range);
        \\    }
        \\
        \\    pub fn zero_key(value: u256) {
        \\        let slot: StorageSlot = @storageDerive("vault.root");
        \\        @storageWordStore(slot, 0, value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const write_fn = try functionSlice(rendered, "write");
    const read_fn = try functionSlice(rendered, "read");
    const clear_fn = try functionSlice(rendered, "clear");
    const zero_key_fn = try functionSlice(rendered, "zero_key");

    try testing.expect(std.mem.containsAtLeast(u8, write_fn, 1, "keccak256"));
    try testing.expect(std.mem.containsAtLeast(u8, write_fn, 1, "sstore"));
    try testing.expect(std.mem.containsAtLeast(u8, read_fn, 1, "keccak256"));
    try testing.expect(std.mem.containsAtLeast(u8, read_fn, 1, "sload"));
    try testing.expect(std.mem.containsAtLeast(u8, clear_fn, 1, "keccak256"));
    try testing.expect(std.mem.containsAtLeast(u8, clear_fn, 1, "sstore"));
    try testing.expect(std.mem.containsAtLeast(u8, clear_fn, 1, "lt "));
    try testing.expect(std.mem.containsAtLeast(u8, clear_fn, 1, "=>"));
    try testing.expect(std.mem.containsAtLeast(u8, zero_key_fn, 1, "keccak256"));
    try testing.expect(std.mem.containsAtLeast(u8, zero_key_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, zero_key_fn, 3, "mstore256"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.storage.derive"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.storage.word_load"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.storage.word_store"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.storage.range_erase"));
}

test "compiler lowers signed integer operations through signed SIR ops" {
    const source_text =
        \\contract SignedOps {
        \\    pub fn signed_div(a: i256, b: i256) -> i256 {
        \\        return a / b;
        \\    }
        \\
        \\    pub fn signed_mod(a: i256, b: i256) -> i256 {
        \\        return a % b;
        \\    }
        \\
        \\    pub fn signed_shr(a: i256, b: i8) -> i256 {
        \\        return a >> b;
        \\    }
        \\
        \\    pub fn signed_gt(a: i256, b: i256) -> bool {
        \\        return a > b;
        \\    }
        \\
        \\    pub fn signed_checked_add(a: i256, b: i256) -> i256 {
        \\        return a + b;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, try functionSlice(rendered, "signed_div"), 1, "sdiv"));
    try testing.expect(std.mem.containsAtLeast(u8, try functionSlice(rendered, "signed_mod"), 1, "smod"));
    try testing.expect(std.mem.containsAtLeast(u8, try functionSlice(rendered, "signed_shr"), 1, "sar"));
    try testing.expect(std.mem.containsAtLeast(u8, try functionSlice(rendered, "signed_gt"), 1, "sgt"));
    try testing.expect(std.mem.containsAtLeast(u8, try functionSlice(rendered, "signed_checked_add"), 1, "slt"));
}

test "compiler lowers unsigned integer operations through unsigned SIR ops" {
    const source_text =
        \\contract UnsignedOps {
        \\    pub fn unsigned_div(a: u256, b: u256) -> u256 {
        \\        return a / b;
        \\    }
        \\
        \\    pub fn unsigned_mod(a: u256, b: u256) -> u256 {
        \\        return a % b;
        \\    }
        \\
        \\    pub fn unsigned_shr(a: u256, b: u8) -> u256 {
        \\        return a >> b;
        \\    }
        \\
        \\    pub fn unsigned_gt(a: u256, b: u256) -> bool {
        \\        return a > b;
        \\    }
        \\
        \\    pub fn unsigned_checked_add(a: u256, b: u256) -> u256 {
        \\        return a + b;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const div_fn = try functionSlice(rendered, "unsigned_div");
    const mod_fn = try functionSlice(rendered, "unsigned_mod");
    const shr_fn = try functionSlice(rendered, "unsigned_shr");
    const gt_fn = try functionSlice(rendered, "unsigned_gt");
    const add_fn = try functionSlice(rendered, "unsigned_checked_add");

    try testing.expect(std.mem.containsAtLeast(u8, div_fn, 1, " = div "));
    try testing.expect(!std.mem.containsAtLeast(u8, div_fn, 1, " = sdiv "));
    try testing.expect(std.mem.containsAtLeast(u8, mod_fn, 1, " = mod "));
    try testing.expect(!std.mem.containsAtLeast(u8, mod_fn, 1, " = smod "));
    try testing.expect(std.mem.containsAtLeast(u8, shr_fn, 1, " = shr "));
    try testing.expect(!std.mem.containsAtLeast(u8, shr_fn, 1, " = sar "));
    try testing.expect(std.mem.containsAtLeast(u8, gt_fn, 1, " = gt "));
    try testing.expect(!std.mem.containsAtLeast(u8, gt_fn, 1, " = sgt "));
    try testing.expect(std.mem.containsAtLeast(u8, add_fn, 1, " = lt "));
    try testing.expect(!std.mem.containsAtLeast(u8, add_fn, 1, " = slt "));
}

test "compiler lowers generic requires clauses with substituted integer signedness" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract GenericRequires {
        \\    fn add(comptime T: type, a: T, b: T) -> T
        \\        requires a <= std.constants.U256_MAX - b
        \\    {
        \\        return a + b;
        \\    }
        \\
        \\    fn guarded_div(comptime T: type, a: T, b: T) -> T
        \\        requires a >= b
        \\    {
        \\        return a / b;
        \\    }
        \\
        \\    pub fn run_unsigned(a: u256, b: u256) -> u256 {
        \\        return add(u256, a, b);
        \\    }
        \\
        \\    pub fn run_signed(a: i256, b: i256) -> i256 {
        \\        return guarded_div(i256, a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn add__u256"));
    const signed_fn = try functionSlice(rendered, "guarded_div__i256");
    try testing.expect(std.mem.containsAtLeast(u8, signed_fn, 1, "sdiv"));
}

test "frontend reuses storage roots for compound map assignments without parent handle rewrites" {
    const source_text =
        \\contract MapCounter {
        \\    storage var balances: map<address, u256>;
        \\    storage var allowances: map<address, map<address, u256>>;
        \\
        \\    pub fn add_balance(owner: address, amount: u256) {
        \\        balances[owner] += amount;
        \\    }
        \\
        \\    pub fn add_allowance(owner: address, spender: address, amount: u256) {
        \\        allowances[owner][spender] += amount;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "ora.sload \"balances\""));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "ora.sload \"allowances\""));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "ora.map_store"));
}

test "compiler lowers dynamic byte concat and slice through OraToSIR" {
    const source_text =
        \\pub fn join(a: bytes, b: bytes) -> bytes {
        \\    return a + b;
        \\}
        \\
        \\pub fn cut(data: bytes, start: u256, length: u256) -> bytes {
        \\    return @slice(data, start, length);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 3, "mcopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.concat"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.slice"));
}
