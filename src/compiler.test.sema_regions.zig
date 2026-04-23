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

test "old ghost storage expressions are not const-folded away" {
    const source_text =
        \\comptime const std = @import("std");
        \\contract Mini {
        \\    ghost storage var transactionCount: u256 = 0;
        \\
        \\    pub fn deposit()
        \\        requires transactionCount < std.constants.U256_MAX
        \\        ensures transactionCount == old(transactionCount) + 1
        \\    {
        \\        transactionCount = transactionCount + 1;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "ora.old") != null);

    var summary = try verifyTextWithoutDegradation(source_text, "deposit");
    defer summary.deinit(testing.allocator);
    try testing.expect(summary.success);
    try testing.expectEqual(@as(usize, 0), summary.errors_len);
    try testing.expectEqual(false, summary.degraded);
}

test "compiler rejects storage write after extern call on same slot" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\    storage var other: u256;
        \\
        \\    pub fn bad(to: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).transfer(to, balance);
        \\        _ = call_result;
        \\        balance = 2;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write storage slot 'balance' after external call because it was written before the call"));
}

test "compiler allows writes to different storage slots around extern call" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\    storage var other: u256;
        \\
        \\    pub fn ok(to: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).transfer(to, balance);
        \\        _ = call_result;
        \\        other = 2;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler allows writes around extern staticcall" {
    const source_text =
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\
        \\    pub fn ok(user: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).balanceOf(user);
        \\        _ = call_result;
        \\        balance = 2;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler allows post-call writes when pre-call storage write is branch-local" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Test {
        \\    storage var balance: u256 = 0;
        \\    storage var token: address;
        \\
        \\    pub fn example(flag: bool, addr: address) {
        \\        if (flag) {
        \\            balance = 100;
        \\        }
        \\        let ok = external<ERC20>(token, gas: 50000).transfer(addr, 1);
        \\        _ = ok;
        \\        balance += 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler rejects post-call writes when all branches wrote same storage slot before call" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Test {
        \\    storage var balance: u256 = 0;
        \\    storage var token: address;
        \\
        \\    pub fn example(flag: bool, addr: address) {
        \\        if (flag) {
        \\            balance = 100;
        \\        } else {
        \\            balance = 200;
        \\        }
        \\        let ok = external<ERC20>(token, gas: 50000).transfer(addr, 1);
        \\        _ = ok;
        \\        balance += 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write storage slot 'balance' after external call because it was written before the call"));
}

test "compiler still rejects same-slot write before and after extern call without branches" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Test {
        \\    storage var balance: u256 = 0;
        \\    storage var token: address;
        \\
        \\    pub fn example(addr: address) {
        \\        balance = 100;
        \\        let ok = external<ERC20>(token, gas: 50000).transfer(addr, 1);
        \\        _ = ok;
        \\        balance += 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write storage slot 'balance' after external call because it was written before the call"));
}

test "compiler lowers shorthand storage field assignments" {
    const source_text =
        \\contract Wallet {
        \\    storage owner: address;
        \\
        \\    pub fn setOwner(next: address) {
        \\        owner = next;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
}

test "compiler tracks per-function read and write effects" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\    tstore var pending: u256;
        \\
        \\    pub fn read_only() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    pub fn write_only(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn mixed(value: u256) -> u256 {
        \\        pending += value;
        \\        return total;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);

    const read_only = item_index.lookup("read_only").?;
    const write_only = item_index.lookup("write_only").?;
    const mixed = item_index.lookup("mixed").?;

    switch (typecheck.itemEffect(read_only)) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(write_only)) {
        .writes => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(mixed)) {
        .reads_writes => |effect| {
            try testing.expect(containsEffectSlot(effect.reads, "pending", .transient));
            try testing.expect(containsEffectSlot(effect.reads, "total", .storage));
            try testing.expect(containsEffectSlot(effect.writes, "pending", .transient));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler composes callee effects into caller summaries" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\    tstore var pending: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    fn write_pending(value: u256) {
        \\        pending = value;
        \\    }
        \\
        \\    pub fn wrapper(value: u256) -> u256 {
        \\        write_pending(value);
        \\        return (read_total());
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const wrapper = item_index.lookup("wrapper").?;

    switch (typecheck.itemEffect(wrapper)) {
        .reads_writes => |effect| {
            try testing.expect(containsEffectSlot(effect.reads, "total", .storage));
            try testing.expect(containsEffectSlot(effect.writes, "pending", .transient));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks keyed map effects by parameter" {
    const source_text =
        \\contract Effects {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn read_balance(user: address) -> u256 {
        \\        return balances[user];
        \\    }
        \\
        \\    pub fn write_balance(user: address, value: u256) {
        \\        balances[user] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const read_balance = item_index.lookup("read_balance").?;
    const write_balance = item_index.lookup("write_balance").?;
    const user_key = [_]compiler.sema.KeySegment{.{ .parameter = 0 }};

    switch (typecheck.itemEffect(read_balance)) {
        .reads => |effect| try testing.expect(containsKeyedEffectSlot(effect.slots, "balances", .storage, &user_key)),
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(write_balance)) {
        .reads_writes => |effect| {
            try testing.expect(containsKeyedEffectSlot(effect.reads, "balances", .storage, &user_key));
            try testing.expect(containsKeyedEffectSlot(effect.writes, "balances", .storage, &user_key));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler rejects direct writes to locked slots" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn write_while_locked(value: u256) {
        \\        @lock(total);
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler rejects keyed writes to the same locked map entry" {
    const source_text =
        \\contract Locked {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn guarded(user: address, value: u256) {
        \\        @lock(balances[user]);
        \\        balances[user] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'balances'"));
}

test "compiler allows callee writes under lock so runtime guards can enforce them" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    fn write_total(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn write_while_locked(value: u256) {
        \\        @lock(total);
        \\        write_total(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler allows helper writes to differently keyed locked array roots" {
    const source_text =
        \\contract Locked {
        \\    storage history: [u256; 8];
        \\
        \\    fn write_history(index: u256, value: u256) {
        \\        history[index] = value;
        \\    }
        \\
        \\    pub fn guarded(locked_index: u256, target_index: u256, value: u256) {
        \\        @lock(history[locked_index]);
        \\        write_history(target_index, value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'history'"));
}

test "compiler allows writes after unlock" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn write_after_unlock(value: u256) {
        \\        @lock(total);
        \\        @unlock(total);
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler allows writes after a conditional lock on only one path" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn guarded(flag: bool, value: u256) {
        \\        if (flag) {
        \\            @lock(total);
        \\        }
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler allows writes after a conditional map-entry lock on only one path" {
    const source_text =
        \\contract Locked {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn guarded(flag: bool, user: address, value: u256) {
        \\        if (flag) {
        \\            @lock(balances[user]);
        \\        }
        \\        balances[user] = value;
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

test "compiler rejects writes when all branches keep a slot locked" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn guarded(flag: bool, value: u256) {
        \\        if (flag) {
        \\            @lock(total);
        \\        } else {
        \\            @lock(total);
        \\        }
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler rejects writes to locked transient slots" {
    const source_text =
        \\contract Locked {
        \\    tstore var pending: u256;
        \\
        \\    pub fn write_while_locked(value: u256) {
        \\        @lock(pending);
        \\        pending = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked transient slot 'pending'"));
}

test "compiler composes contract member call effects into caller summaries" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\}
        \\
        \\pub fn wrapper(vault: Vault) -> u256 {
        \\    return vault.read_total();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    switch (typecheck.itemEffect(ast_file.root_items[1])) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler allows locked writes through contract member calls for runtime guarding" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256;
        \\
        \\    fn write_total(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn guarded(vault: Vault, value: u256) {
        \\        @lock(total);
        \\        vault.write_total(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler composes effects through local function aliases" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    pub fn wrapper() -> u256 {
        \\        let reader = read_total;
        \\        return reader();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const wrapper = item_index.lookup("wrapper").?;

    switch (typecheck.itemEffect(wrapper)) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks per-expression composed call effects" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    pub fn wrapper() -> u256 {
        \\        return read_total();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const wrapper = ast_file.item(contract.members[2]).Function;
    const ret_stmt = ast_file.statement(ast_file.body(wrapper.body).statements[0]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    switch (typecheck.exprEffect(ret_stmt.value.?)) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler allows locked writes through local function aliases for runtime guarding" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    fn write_total(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn guarded(value: u256) {
        \\        let writer = write_total;
        \\        @lock(total);
        \\        writer(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler composes effects through member-derived function aliases" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\}
        \\
        \\pub fn wrapper(vault: Vault) -> u256 {
        \\    let reader = vault.read_total;
        \\    return reader();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    switch (typecheck.itemEffect(ast_file.root_items[1])) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks log and havoc effect kinds" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    pub fn noisy(value: u256) {
        \\        log Transfer(value);
        \\        havoc total;
        \\    }
        \\
        \\    pub fn wrapper(value: u256) {
        \\        noisy(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const noisy = item_index.lookup("noisy").?;
    const wrapper = item_index.lookup("wrapper").?;

    switch (typecheck.itemEffect(noisy)) {
        .side_effects => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
        },
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(wrapper)) {
        .side_effects => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
        },
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks lock and unlock effect kinds" {
    const source_text =
        \\contract Locks {
        \\    storage total: u256;
        \\
        \\    pub fn guarded() {
        \\        @lock(total);
        \\        @unlock(total);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const guarded = item_index.lookup("guarded").?;

    switch (typecheck.itemEffect(guarded)) {
        .side_effects => |effect| {
            try testing.expect(effect.has_lock);
            try testing.expect(effect.has_unlock);
        },
        .reads => |effect| {
            try testing.expect(effect.has_lock);
            try testing.expect(effect.has_unlock);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_lock);
            try testing.expect(effect.has_unlock);
            try testing.expect(containsEffectSlot(effect.reads, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler composes effects to a fixpoint across mutual recursion" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    fn ping(n: u256) {
        \\        if (n == 0) {
        \\            return;
        \\        }
        \\        log Ping(n);
        \\        pong(n - 1);
        \\    }
        \\
        \\    fn pong(n: u256) {
        \\        if (n == 0) {
        \\            return;
        \\        }
        \\        total = n;
        \\        ping(n - 1);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const ping = item_index.lookup("ping").?;
    const pong = item_index.lookup("pong").?;

    switch (typecheck.itemEffect(ping)) {
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(pong)) {
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks declaration root regions in type check output" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\    memory var scratch: u256;
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn inspect(value: u256) -> u256 {
        \\    let local = value;
        \\    return local;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    try testing.expectEqual(compiler.sema.Region.storage, typecheck.itemLocatedType(contract.members[0]).region);
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.itemLocatedType(contract.members[1]).region);
    try testing.expectEqual(compiler.sema.Region.transient, typecheck.itemLocatedType(contract.members[2]).region);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const parameter_type = typecheck.pattern_types[function.parameters[0].pattern.index()];
    try testing.expectEqual(compiler.sema.Region.memory, parameter_type.region);
    try testing.expectEqual(compiler.sema.Provenance.calldata, parameter_type.provenance);

    const body = ast_file.body(function.body);
    const local_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const local_type = typecheck.pattern_types[local_stmt.pattern.index()];
    try testing.expectEqual(compiler.sema.Region.memory, local_type.region);
    try testing.expectEqual(compiler.sema.Provenance.calldata, local_type.provenance);
}

test "compiler allows implicit region reads into locals" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn inspect(value: u256) -> u256 {
        \\    let from_param = value;
        \\    let from_storage = total;
        \\    let from_tstore = pending;
        \\    return from_param + from_storage + from_tstore;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler rejects writes to calldata and direct storage transient transfer" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn inspect(value: u256) -> u256 {
        \\    value = total;
        \\    pending = total;
        \\    total = pending;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(diags.len() >= 2);
    try testing.expect(diagnosticMessagesContain(diags, "assignment expects region 'transient'"));
    try testing.expect(diagnosticMessagesContain(diags, "assignment expects region 'storage'"));
}

test "compiler lowers labeled block statements through syntax AST path" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    outer: {
        \\        let value = 1;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const labeled = ast_file.statement(body.statements[0]).LabeledBlock;

    try testing.expectEqualStrings("outer", labeled.label);
    try testing.expectEqual(@as(usize, 1), ast_file.body(labeled.body).statements.len);
}

test "compiler lowers lock and unlock statements through syntax AST and HIR paths" {
    const source_text =
        \\contract Vault {
        \\    storage balances: u256;
        \\
        \\    pub fn run() {
        \\        @lock(balances);
        \\        @unlock(balances);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const function = ast_file.item(contract.members[1]).Function;
    const body = ast_file.body(function.body);
    try testing.expect(ast_file.statement(body.statements[0]).* == .Lock);
    try testing.expect(ast_file.statement(body.statements[1]).* == .Unlock);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock_placeholder"));
}

test "compiler emits tstore guard before guarded storage writes" {
    const source_text =
        \\contract GuardedWrites {
        \\    storage balances: u256;
        \\
        \\    pub fn touch() {
        \\        @lock(balances);
        \\        balances = 1;
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "guarded-write.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tstore.guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
}

test "compiler emits keyed tstore guard before guarded indexed storage writes" {
    const source_text =
        \\contract GuardedWrites {
        \\    storage history: [u256; 8];
        \\
        \\    fn write_history(index: u256, value: u256) {
        \\        history[index] = value;
        \\    }
        \\
        \\    pub fn touch(index: u256, value: u256) {
        \\        @lock(history[index]);
        \\        write_history(index, value);
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "guarded-indexed-write.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tstore.guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"history[]\""));
}

test "compiler lowers grouped lock paths through real lock ops" {
    const source_text =
        \\contract Vault {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn run(user: address) {
        \\        @lock((balances[user]));
        \\        @unlock((balances[user]));
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock_placeholder"));
}

test "compiler rejects lock and unlock builtins in expression position" {
    const source_text =
        \\contract Locked {
        \\    storage var balances: map<address, u256>;
        \\
        \\    pub fn bad_lock(user: address) -> u256 {
        \\        let tmp = @lock(balances[user]);
        \\        return tmp;
        \\    }
        \\
        \\    pub fn bad_unlock(user: address) -> bool {
        \\        @lock(balances[user]);
        \\        return @unlock(balances[user]);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "@lock is statement-only and cannot be used in expression position"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "@unlock is statement-only and cannot be used in expression position"));
}

test "compiler does not partially evaluate storage-reading helper calls inside requires" {
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

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "call @dep") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "arith.cmpi ule, %arg0, %c0_i256_1") == null);
}

test "compiler lowers real HIR if and try regions" {
    const source_text =
        \\log Ping(value: u256);
        \\
        \\pub fn flow(ok: bool, next: u256) -> u256 {
        \\    let value = next;
        \\    if (ok) {
        \\        log Ping(next);
        \\    } else {
        \\        assume(next >= 0);
        \\    }
        \\    try {
        \\        assert(ok);
        \\    } catch (err) {
        \\        havoc err;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.log"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assume"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.havoc"));
}

test "compiler lowers real HIR if regions with carried locals" {
    const source_text =
        \\pub fn choose(ok: bool, start: u256) -> u256 {
        \\    let value = start;
        \\    if (ok) {
        \\        value = value + 1;
        \\    } else {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler does not emit invalid ora.conditional_return inside deferred scf regions" {
    const source_text =
        \\pub fn choose(level: u256) -> u256 {
        \\    if (level > 10) {
        \\        return 0;
        \\    } else {
        \\        if (level > 5) {
        \\            return 1;
        \\        } else {
        \\            if (level > 2) {
        \\                return 2;
        \\            } else {
        \\                return 3;
        \\            }
        \\        }
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler lowers real HIR try regions with carried locals" {
    const source_text =
        \\pub fn recover(start: u256) -> u256 {
        \\    let value = start;
        \\    try {
        \\        value = value + 1;
        \\    } catch (err) {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
}

test "compiler lowers real HIR switch regions with carried locals" {
    const source_text =
        \\pub fn choose(tag: u256, start: u256) -> u256 {
        \\    let value = start;
        \\    switch (tag) {
        \\        0 => {
        \\            value = value + 1;
        \\        },
        \\        else => {
        \\            value = value + 2;
        \\        }
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler lowers real HIR switch regions" {
    const source_text =
        \\pub fn classify(v: u256, fallback: u256) -> u256 {
        \\    switch (v) {
        \\        0 => {
        \\            assert(true);
        \\        },
        \\        1...2 => {
        \\            assume(v >= 1);
        \\        },
        \\        else => {
        \\            havoc fallback;
        \\        }
        \\    }
        \\    return v;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assume"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.havoc"));
}

test "compiler lowers real HIR while loops for storage-driven loops" {
    const source_text =
        \\storage count: u256;
        \\
        \\pub fn drain(limit: u256) {
        \\    while (count < limit) {
        \\        count = count + 1;
        \\        assert(count >= 1);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.condition"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sload"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler tracks storage_class on variable declarations" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() {
        \\    storage var x: u256 = 0;
        \\    memory var y: u256 = 0;
        \\    let z: u256 = 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);

    const decl_x = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.storage, typecheck.pattern_types[decl_x.pattern.index()].region);

    const decl_y = ast_file.statement(body.statements[1]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_y.pattern.index()].region);

    const decl_z = ast_file.statement(body.statements[2]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_z.pattern.index()].region);
}

test "compiler rejects region-incompatible variable initialization" {
    const source_text =
        \\contract Vault {
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn example() {
        \\    storage var x: u256 = pending;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(!diags.isEmpty());
    try testing.expect(diagnosticMessagesContain(diags, "declaration expects region 'storage', found 'transient'"));
}

test "compiler rejects region-incompatible field initializer" {
    const source_text =
        \\contract Vault {
        \\    tstore var pending: u256;
        \\    storage var committed: u256 = pending;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(!diags.isEmpty());
    try testing.expect(diagnosticMessagesContain(diags, "field 'committed' expects region"));
}

test "compiler allows passing storage values to function parameters" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn helper(x: u256) -> u256 {
        \\    return x;
        \\}
        \\
        \\pub fn example() -> u256 {
        \\    return helper(total);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler allows returning storage values from functions" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() -> u256 {
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler tracks storage_class on inferred-type variable declarations" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() {
        \\    storage var x = 0;
        \\    memory var y = 0;
        \\    tstore var z = 0;
        \\    let w = 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);

    const decl_x = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.storage, typecheck.pattern_types[decl_x.pattern.index()].region);

    const decl_y = ast_file.statement(body.statements[1]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_y.pattern.index()].region);

    const decl_z = ast_file.statement(body.statements[2]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.transient, typecheck.pattern_types[decl_z.pattern.index()].region);

    const decl_w = ast_file.statement(body.statements[3]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_w.pattern.index()].region);
}

test "compiler tracks tstore var inside function body" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() {
        \\    tstore var temp: u256 = 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);

    const decl = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.transient, typecheck.pattern_types[decl.pattern.index()].region);
}

