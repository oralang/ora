const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;
const mlir = @import("mlir_c_api").c;
const mlir_cfg = @import("mlir/cfg.zig");
const runtime_checks = @import("mlir/runtime_checks.zig");
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

fn extractSirGlobalSlotsJson(context: mlir.MlirContext, module: mlir.MlirModule) ![]u8 {
    const slots_ref = mlir.oraExtractSIRGlobalSlots(context, module);
    defer if (slots_ref.data != null) mlir.oraStringRefFree(slots_ref);
    if (slots_ref.data == null or slots_ref.length == 0) return error.TestUnexpectedResult;
    return try testing.allocator.dupe(u8, slots_ref.data[0..slots_ref.length]);
}

fn expectGlobalSlot(slots_json: []const u8, name: []const u8, expected: i128) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, slots_json, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    const actual = parsed.value.object.get(name) orelse return error.TestUnexpectedResult;
    try testing.expect(actual == .integer);
    try testing.expectEqual(expected, actual.integer);
}

fn createOraMlirContext() mlir.MlirContext {
    const ctx = mlir.oraContextCreate();
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
    mlir.oraContextLoadSIRDialect(ctx);
    _ = mlir.oraDialectRegister(ctx);
    return ctx;
}

fn parseOraModule(ctx: mlir.MlirContext, text: []const u8) !mlir.MlirModule {
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    if (mlir.oraModuleIsNull(module)) return error.TestUnexpectedResult;
    return module;
}

fn printModuleTextForTest(module: mlir.MlirModule) ![]u8 {
    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    if (module_text_ref.data == null or module_text_ref.length == 0) return error.TestUnexpectedResult;
    return testing.allocator.dupe(u8, module_text_ref.data[0..module_text_ref.length]);
}

fn setModuleBoolAttr(ctx: mlir.MlirContext, module: mlir.MlirModule, name: []const u8) void {
    const attr = mlir.oraBoolAttrCreate(ctx, true);
    mlir.oraOperationSetAttributeByName(
        mlir.oraModuleGetOperation(module),
        mlir.oraStringRefCreate(name.ptr, name.len),
        attr,
    );
}

test "compiler removes simple source inline helper calls before SIR call conversion" {
    const source_text =
        \\contract InlineLowering {
        \\    inline fn identity(value: u256) -> u256 {
        \\        return value;
        \\    }
        \\
        \\    pub fn run(value: u256) -> u256 {
        \\        return identity(value);
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

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @identity"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler binds source inline arguments once before SIR call conversion" {
    const source_text =
        \\contract InlineLowering {
        \\    storage var saved: u256;
        \\
        \\    inline fn observeTwice(value: u256) -> u256 {
        \\        if (value == value) {
        \\            return 14;
        \\        }
        \\        return 0;
        \\    }
        \\
        \\    fn sideEffectArg() -> u256 {
        \\        saved = 1;
        \\        return 7;
        \\    }
        \\
        \\    pub fn run() -> u256 {
        \\        return observeTwice(sideEffectArg());
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @observeTwice"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, run_fn, "icall @sideEffectArg"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands early-return source inline helper calls before SIR call conversion" {
    const source_text =
        \\contract InlineLowering {
        \\    inline fn choose(flag: bool, value: u256) -> u256 {
        \\        if (flag) {
        \\            return value;
        \\        }
        \\        return 0;
        \\    }
        \\
        \\    pub fn run(flag: bool, value: u256) -> u256 {
        \\        return choose(flag, value);
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

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @choose"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands nested early-return source inline helper calls before SIR call conversion" {
    const source_text =
        \\contract InlineLowering {
        \\    inline fn choose(flag: bool, other: bool, value: u256) -> u256 {
        \\        if (flag) {
        \\            if (other) {
        \\                return value;
        \\            }
        \\            return 1;
        \\        }
        \\        return 0;
        \\    }
        \\
        \\    pub fn run(flag: bool, other: bool, value: u256) -> u256 {
        \\        return choose(flag, other, value);
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

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @choose"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands Result source inline helper calls before SIR call conversion" {
    const source_text =
        \\error Failure;
        \\
        \\contract InlineLowering {
        \\    inline fn pick(flag: bool, value: u256) -> Result<u256, Failure> {
        \\        if (flag) {
        \\            return Ok(value);
        \\        }
        \\        return Err(Failure());
        \\    }
        \\
        \\    pub fn run(flag: bool, value: u256) -> Result<u256, Failure> {
        \\        return pick(flag, value);
        \\    }
        \\
        \\    pub fn unwrap(flag: bool, value: u256) -> !u256 | Failure {
        \\        let out: u256 = try pick(flag, value);
        \\        return out;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @pick"));

    const unwrap_fn = try functionSlice(rendered, "unwrap");
    try testing.expect(!std.mem.containsAtLeast(u8, unwrap_fn, 1, "icall @pick"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands multiple fallible source inline early returns before SIR call conversion" {
    const source_text =
        \\error Failure;
        \\
        \\contract InlineLowering {
        \\    inline fn check(first: bool, second: bool) -> !u256 | Failure {
        \\        if (first) {
        \\            return Failure;
        \\        }
        \\        if (second) {
        \\            return Failure;
        \\        }
        \\        return 7;
        \\    }
        \\
        \\    pub fn run(first: bool, second: bool) -> !u256 | Failure {
        \\        let out: u256 = try check(first, second);
        \\        return out;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @check"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands unsigned checked-prelude source inline helpers before SIR call conversion" {
    const source_text =
        \\comptime const constants = @import("std/constants");
        \\
        \\error Overflow;
        \\
        \\contract InlineLowering {
        \\    inline fn checkedAdd(current: u256, delta: u256) -> !u256 | Overflow {
        \\        if (current > constants.U256_MAX - delta) {
        \\            return Overflow;
        \\        }
        \\        return current + delta;
        \\    }
        \\
        \\    pub fn run(current: u256, delta: u256) -> !u256 | Overflow {
        \\        let out: u256 = try checkedAdd(current, delta);
        \\        return out;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @checkedAdd"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands signed checked-prelude source inline helpers before SIR call conversion" {
    const source_text =
        \\comptime const constants = @import("std/constants");
        \\
        \\error Overflow;
        \\
        \\contract InlineLowering {
        \\    inline fn checkedSignedAdd(current: i256, delta: i256) -> !i256 | Overflow {
        \\        if (delta > 0 && current > constants.I256_MAX - delta) {
        \\            return Overflow;
        \\        }
        \\        if (delta < 0 && current < constants.I256_MIN - delta) {
        \\            return Overflow;
        \\        }
        \\        return current + delta;
        \\    }
        \\
        \\    pub fn run(current: i256, delta: i256) -> !i256 | Overflow {
        \\        let out: i256 = try checkedSignedAdd(current, delta);
        \\        return out;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @checkedSignedAdd"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands void early-return source inline helper calls before SIR call conversion" {
    const source_text =
        \\contract InlineLowering {
        \\    storage var saved: u256;
        \\
        \\    inline fn storeUnless(flag: bool, value: u256) {
        \\        if (flag) {
        \\            return;
        \\        }
        \\        saved = value;
        \\    }
        \\
        \\    pub fn run(flag: bool, value: u256) {
        \\        storeUnless(flag, value);
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

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @storeUnless"));
    try testing.expect(std.mem.containsAtLeast(u8, run_fn, 1, "sstore"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands source inline helpers with comptime integer parameters" {
    const source_text =
        \\contract InlineLowering {
        \\    inline fn chooseMode(comptime mode: u256) -> u256 {
        \\        var out: u256 = 0;
        \\        comptime {
        \\            if (mode == 1) {
        \\                out = 7;
        \\            } else {
        \\                out = 9;
        \\            }
        \\        }
        \\        return out;
        \\    }
        \\
        \\    pub fn one() -> u256 {
        \\        return chooseMode(1);
        \\    }
        \\
        \\    pub fn two() -> u256 {
        \\        return chooseMode(2);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const one_fn = try functionSlice(rendered, "one");
    try testing.expect(!std.mem.containsAtLeast(u8, one_fn, 1, "icall @chooseMode"));
    try testing.expect(std.mem.containsAtLeast(u8, one_fn, 1, "const 0x7"));

    const two_fn = try functionSlice(rendered, "two");
    try testing.expect(!std.mem.containsAtLeast(u8, two_fn, 1, "icall @chooseMode"));
    try testing.expect(std.mem.containsAtLeast(u8, two_fn, 1, "const 0x9"));

    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands source inline helpers with callsite-known runtime literals" {
    const source_text =
        \\contract InlineLowering {
        \\    inline fn chooseImplicit(mode: u256) -> u256 {
        \\        var out: u256 = 0;
        \\        comptime {
        \\            if (mode == 1) {
        \\                out = 7;
        \\            } else {
        \\                out = 9;
        \\            }
        \\        }
        \\        return out;
        \\    }
        \\
        \\    inline fn chooseBool(flag: bool) -> u256 {
        \\        var out: u256 = 0;
        \\        comptime {
        \\            if (flag) {
        \\                out = 11;
        \\            } else {
        \\                out = 12;
        \\            }
        \\        }
        \\        return out;
        \\    }
        \\
        \\    pub fn one() -> u256 {
        \\        return chooseImplicit(1);
        \\    }
        \\
        \\    pub fn two() -> u256 {
        \\        return chooseImplicit(2);
        \\    }
        \\
        \\    pub fn yes() -> u256 {
        \\        return chooseBool(true);
        \\    }
        \\
        \\    pub fn no() -> u256 {
        \\        return chooseBool(false);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const one_fn = try functionSlice(rendered, "one");
    try testing.expect(!std.mem.containsAtLeast(u8, one_fn, 1, "icall @chooseImplicit"));
    try testing.expect(std.mem.containsAtLeast(u8, one_fn, 1, "const 0x7"));

    const two_fn = try functionSlice(rendered, "two");
    try testing.expect(!std.mem.containsAtLeast(u8, two_fn, 1, "icall @chooseImplicit"));
    try testing.expect(std.mem.containsAtLeast(u8, two_fn, 1, "const 0x9"));

    const yes_fn = try functionSlice(rendered, "yes");
    try testing.expect(!std.mem.containsAtLeast(u8, yes_fn, 1, "icall @chooseBool"));
    try testing.expect(std.mem.containsAtLeast(u8, yes_fn, 1, "const 0xB"));

    const no_fn = try functionSlice(rendered, "no");
    try testing.expect(!std.mem.containsAtLeast(u8, no_fn, 1, "icall @chooseBool"));
    try testing.expect(std.mem.containsAtLeast(u8, no_fn, 1, "const 0xC"));

    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler expands source inline helpers with structured non-returning regions" {
    const source_text =
        \\contract InlineLowering {
        \\    inline fn structuredChoice(flag: bool, tag: u256, value: u256) -> u256 {
        \\        var out: u256 = 0;
        \\        if (flag) {
        \\            out = value;
        \\        } else {
        \\            out = 4;
        \\        }
        \\        switch (tag) {
        \\            1 => {
        \\                out = out;
        \\            }
        \\            else => {
        \\                out = 9;
        \\            }
        \\        }
        \\        return out;
        \\    }
        \\
        \\    inline fn countThree() -> u256 {
        \\        var count: u256 = 0;
        \\        while (count < 3)
        \\            invariant count <= 3
        \\        {
        \\            count = count + 1;
        \\        }
        \\        return count;
        \\    }
        \\
        \\    pub fn run(flag: bool, tag: u256, value: u256) -> u256 {
        \\        return structuredChoice(flag, tag, value);
        \\    }
        \\
        \\    pub fn loopThree() -> u256 {
        \\        return countThree();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const run_fn = try functionSlice(rendered, "run");
    try testing.expect(!std.mem.containsAtLeast(u8, run_fn, 1, "icall @structuredChoice"));

    const loop_three_fn = try functionSlice(rendered, "loopThree");
    try testing.expect(!std.mem.containsAtLeast(u8, loop_three_fn, 1, "icall @countThree"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "OraToSIR fails closed when source inline call survives inlining" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @helper(%arg0: !ora.int<256, false>) -> !ora.int<256, false> attributes {ora.inline = true, ora.source_inline = true, ora.visibility = "private"} {
        \\      cf.br ^next(%arg0 : !ora.int<256, false>)
        \\    ^next(%value: !ora.int<256, false>):
        \\      return %value : !ora.int<256, false>
        \\    }
        \\    func.func @run(%arg0: !ora.int<256, false>) -> !ora.int<256, false> attributes {ora.visibility = "pub"} {
        \\      %0 = func.call @helper(%arg0) : (!ora.int<256, false>) -> !ora.int<256, false>
        \\      return %0 : !ora.int<256, false>
        \\    }
        \\  }
        \\}
    ;

    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(!mlir.oraConvertToSIR(ctx, module, false));
}

test "Ora inline without source guarantee does not fail closed" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @helper(%arg0: !ora.int<256, false>) -> !ora.int<256, false> attributes {ora.inline = true, ora.visibility = "private"} {
        \\      cf.br ^next(%arg0 : !ora.int<256, false>)
        \\    ^next(%value: !ora.int<256, false>):
        \\      return %value : !ora.int<256, false>
        \\    }
        \\    func.func @run(%arg0: !ora.int<256, false>) -> !ora.int<256, false> attributes {ora.visibility = "pub"} {
        \\      %0 = func.call @helper(%arg0) : (!ora.int<256, false>) -> !ora.int<256, false>
        \\      return %0 : !ora.int<256, false>
        \\    }
        \\  }
        \\}
    ;

    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));

    var stats: mlir.OraMlirPassStatisticsC = std.mem.zeroes(mlir.OraMlirPassStatisticsC);
    try testing.expect(mlir.oraCanonicalizeOraMLIRWithStatisticsOut(ctx, module, false, &stats));
    try testing.expectEqual(@as(u64, 0), stats.ora_source_inline_failures);
}

fn functionSlice(sir_text: []const u8, function_name: []const u8) ![]const u8 {
    const header = try std.fmt.allocPrint(testing.allocator, "fn {s}:", .{function_name});
    defer testing.allocator.free(header);
    const start = std.mem.indexOf(u8, sir_text, header) orelse return error.TestUnexpectedResult;
    const search_from = start + header.len;
    const rel_end = std.mem.indexOfPos(u8, sir_text, search_from, "\nfn ");
    const end = rel_end orelse sir_text.len;
    return sir_text[start..end];
}

fn oraFunctionSlice(ora_text: []const u8, function_name: []const u8) ![]const u8 {
    const header = try std.fmt.allocPrint(testing.allocator, "func.func @{s}", .{function_name});
    defer testing.allocator.free(header);
    const start = std.mem.indexOf(u8, ora_text, header) orelse return error.TestUnexpectedResult;
    const search_from = start + header.len;
    const rel_end = std.mem.indexOfPos(u8, ora_text, search_from, "func.func @");
    const end = rel_end orelse ora_text.len;
    return ora_text[start..end];
}

fn countSirBitcastsForSource(
    source_text: []const u8,
    debug_info: bool,
    skip_manual_bitcast_fold: bool,
    run_framework_canonicalizer: bool,
) !usize {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    if (skip_manual_bitcast_fold)
        setModuleBoolAttr(hir_result.context, hir_result.module.raw_module, "ora.phase0.skip_manual_bitcast_fold");
    if (run_framework_canonicalizer)
        setModuleBoolAttr(hir_result.context, hir_result.module.raw_module, "ora.phase0.run_sir_framework_canonicalizer");

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, debug_info));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    return std.mem.count(u8, rendered, "sir.bitcast");
}

fn renderSirTextForSourceWithAttrs(
    source_text: []const u8,
    debug_info: bool,
    skip_manual_bitcast_fold: bool,
    run_framework_canonicalizer: bool,
) ![]u8 {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    if (skip_manual_bitcast_fold)
        setModuleBoolAttr(hir_result.context, hir_result.module.raw_module, "ora.phase0.skip_manual_bitcast_fold");
    if (run_framework_canonicalizer)
        setModuleBoolAttr(hir_result.context, hir_result.module.raw_module, "ora.phase0.run_sir_framework_canonicalizer");

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, debug_info));
    return renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
}

fn expectOrderedNeedles(haystack: []const u8, needles: []const []const u8) !void {
    var cursor: usize = 0;
    for (needles) |needle| {
        const found = std.mem.indexOfPos(u8, haystack, cursor, needle) orelse return error.TestUnexpectedResult;
        cursor = found + needle.len;
    }
}

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

const SirDotNode = struct {
    id: []const u8,
    term: []const u8,
    entry: bool = false,
    is_unreachable: bool = false,
    revert: bool = false,
};

const SirDotEdge = struct {
    src: []const u8,
    dst: []const u8,
    label: ?[]const u8 = null,
    backedge: bool = false,
};

const SirDotGraph = struct {
    nodes: []SirDotNode,
    edges: []SirDotEdge,

    fn deinit(self: SirDotGraph, allocator: std.mem.Allocator) void {
        allocator.free(self.nodes);
        allocator.free(self.edges);
    }

    fn nodeIndex(self: SirDotGraph, id: []const u8) ?usize {
        for (self.nodes, 0..) |node, index| {
            if (std.mem.eql(u8, node.id, id)) return index;
        }
        return null;
    }

    fn countTerm(self: SirDotGraph, term: []const u8) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (std.mem.eql(u8, node.term, term)) count += 1;
        }
        return count;
    }

    fn countEntryNodes(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (node.entry) count += 1;
        }
        return count;
    }

    fn countUnreachableNodes(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (node.is_unreachable) count += 1;
        }
        return count;
    }

    fn countRevertNodes(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (node.revert) count += 1;
        }
        return count;
    }

    fn countBackedges(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.edges) |edge| {
            if (edge.backedge) count += 1;
        }
        return count;
    }

    fn expectAllEdgesReferToKnownNodes(self: SirDotGraph) !void {
        for (self.edges) |edge| {
            try testing.expect(self.nodeIndex(edge.src) != null);
            try testing.expect(self.nodeIndex(edge.dst) != null);
        }
    }

    fn expectRevertNodesHaveNoSuccessors(self: SirDotGraph) !void {
        for (self.nodes) |node| {
            if (!node.revert) continue;
            for (self.edges) |edge| {
                try testing.expect(!std.mem.eql(u8, edge.src, node.id));
            }
        }
    }

    fn expectEveryCondBrHasTrueFalseEdges(self: SirDotGraph) !void {
        for (self.nodes) |node| {
            if (!std.mem.eql(u8, node.term, "sir.cond_br")) continue;
            var outgoing: usize = 0;
            var true_edges: usize = 0;
            var false_edges: usize = 0;
            for (self.edges) |edge| {
                if (!std.mem.eql(u8, edge.src, node.id)) continue;
                outgoing += 1;
                if (edge.label) |label| {
                    if (std.mem.eql(u8, label, "true")) true_edges += 1;
                    if (std.mem.eql(u8, label, "false")) false_edges += 1;
                }
            }
            try testing.expectEqual(@as(usize, 2), outgoing);
            try testing.expectEqual(@as(usize, 1), true_edges);
            try testing.expectEqual(@as(usize, 1), false_edges);
        }
    }
};

fn quotedDotAttr(line: []const u8, name: []const u8) ?[]const u8 {
    const attr_start = std.mem.indexOf(u8, line, name) orelse return null;
    var cursor = attr_start + name.len;
    if (cursor >= line.len or line[cursor] != '=') return null;
    cursor += 1;
    if (cursor >= line.len or line[cursor] != '"') return null;
    cursor += 1;
    const value_start = cursor;
    while (cursor < line.len) : (cursor += 1) {
        if (line[cursor] == '"' and (cursor == value_start or line[cursor - 1] != '\\')) {
            return line[value_start..cursor];
        }
    }
    return null;
}

fn nodeTermFromLine(line: []const u8) []const u8 {
    const term_start = std.mem.indexOf(u8, line, "term=") orelse return "";
    const value_start = term_start + "term=".len;
    const rest = line[value_start..];
    if (std.mem.indexOfScalar(u8, rest, '\\')) |end| return rest[0..end];
    if (std.mem.indexOfScalar(u8, rest, '"')) |end| return rest[0..end];
    return rest;
}

fn parseSirDotGraph(allocator: std.mem.Allocator, dot: []const u8) !SirDotGraph {
    var nodes: std.ArrayList(SirDotNode) = .empty;
    errdefer nodes.deinit(allocator);
    var edges: std.ArrayList(SirDotEdge) = .empty;
    errdefer edges.deinit(allocator);

    var lines = std.mem.splitScalar(u8, dot, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (!std.mem.startsWith(u8, line, "f") or std.mem.indexOf(u8, line, "_bb") == null)
            continue;

        if (std.mem.indexOf(u8, line, " -> ")) |arrow| {
            const src = std.mem.trim(u8, line[0..arrow], " \t");
            const after_arrow = line[arrow + " -> ".len ..];
            const bracket = std.mem.indexOf(u8, after_arrow, " [") orelse return error.TestUnexpectedResult;
            const dst = std.mem.trim(u8, after_arrow[0..bracket], " \t");
            try edges.append(allocator, .{
                .src = src,
                .dst = dst,
                .label = quotedDotAttr(line, "label"),
                .backedge = std.mem.containsAtLeast(u8, line, 1, "backedge=\"true\""),
            });
            continue;
        }

        const bracket = std.mem.indexOf(u8, line, " [") orelse continue;
        const id = std.mem.trim(u8, line[0..bracket], " \t");
        try nodes.append(allocator, .{
            .id = id,
            .term = nodeTermFromLine(line),
            .entry = std.mem.containsAtLeast(u8, line, 1, "entry=\"true\""),
            .is_unreachable = std.mem.containsAtLeast(u8, line, 1, "unreachable=\"true\""),
            .revert = std.mem.containsAtLeast(u8, line, 1, "revert=\"true\""),
        });
    }

    return .{
        .nodes = try nodes.toOwnedSlice(allocator),
        .edges = try edges.toOwnedSlice(allocator),
    };
}

test "compiler generates deterministic true SIR branch CFG" {
    const source_text =
        \\pub fn choose(x: u256) -> u256 {
        \\    if (x != 0) {
        \\        return x;
        \\    }
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    const module_before = try printModuleTextForTest(hir_result.module.raw_module);
    defer testing.allocator.free(module_before);

    const dot_a = try mlir_cfg.generateCFG(hir_result.context, hir_result.module.raw_module, testing.allocator, .{ .mode = .sir });
    defer testing.allocator.free(dot_a);
    const dot_b = try mlir_cfg.generateCFG(hir_result.context, hir_result.module.raw_module, testing.allocator, .{ .mode = .sir });
    defer testing.allocator.free(dot_b);
    const module_after = try printModuleTextForTest(hir_result.module.raw_module);
    defer testing.allocator.free(module_after);

    try testing.expectEqualStrings(dot_a, dot_b);
    try testing.expectEqualStrings(module_before, module_after);
    const graph = try parseSirDotGraph(testing.allocator, dot_a);
    defer graph.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), graph.countEntryNodes());
    try testing.expectEqual(@as(usize, 0), graph.countUnreachableNodes());
    try testing.expectEqual(@as(usize, 1), graph.countTerm("sir.cond_br"));
    try graph.expectAllEdgesReferToKnownNodes();
    try graph.expectEveryCondBrHasTrueFalseEdges();
    try graph.expectRevertNodesHaveNoSuccessors();
}

test "compiler generates stable per-function SIR CFGs" {
    const source_text =
        \\pub fn first(x: u256) -> u256 {
        \\    if (x != 0) {
        \\        return x;
        \\    }
        \\    return 1;
        \\}
        \\
        \\pub fn second() -> u256 {
        \\    return 2;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const graphs = try mlir_cfg.generateFunctionCFGs(hir_result.context, hir_result.module.raw_module, testing.allocator, .{ .mode = .sir });
    defer {
        for (graphs) |graph| graph.deinit(testing.allocator);
        testing.allocator.free(graphs);
    }

    try testing.expectEqual(@as(usize, 2), graphs.len);
    try testing.expectEqualStrings("first", graphs[0].name);
    try testing.expectEqualStrings("second", graphs[1].name);
    const first_graph = try parseSirDotGraph(testing.allocator, graphs[0].dot);
    defer first_graph.deinit(testing.allocator);
    const second_graph = try parseSirDotGraph(testing.allocator, graphs[1].dot);
    defer second_graph.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), first_graph.countEntryNodes());
    try testing.expectEqual(@as(usize, 1), second_graph.countEntryNodes());
    try testing.expectEqual(@as(usize, 1), first_graph.countTerm("sir.cond_br"));
    try testing.expectEqual(@as(usize, 0), second_graph.countTerm("sir.cond_br"));
    try testing.expectEqual(@as(usize, 0), second_graph.edges.len);
    try first_graph.expectAllEdgesReferToKnownNodes();
    try first_graph.expectEveryCondBrHasTrueFalseEdges();
    try second_graph.expectAllEdgesReferToKnownNodes();
}

test "compiler marks loop backedges in SIR CFG" {
    const source_text =
        \\pub fn count(n: u256) -> u256 {
        \\    var i: u256 = 0;
        \\    while (i < n) {
        \\        i = i + 1;
        \\    }
        \\    return i;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const dot = try mlir_cfg.generateCFG(hir_result.context, hir_result.module.raw_module, testing.allocator, .{ .mode = .sir });
    defer testing.allocator.free(dot);

    const graph = try parseSirDotGraph(testing.allocator, dot);
    defer graph.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), graph.countEntryNodes());
    try testing.expect(graph.nodes.len >= 3);
    try testing.expect(graph.edges.len >= 2);
    try testing.expect(graph.countBackedges() >= 1);
    try testing.expect(graph.countTerm("sir.cond_br") >= 1);
    try graph.expectAllEdgesReferToKnownNodes();
    try graph.expectEveryCondBrHasTrueFalseEdges();
}

test "compiler SIR CFG marks revert and unreachable blocks without mutating module" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @control(%arg0: !sir.u256) {
        \\    %zero = sir.const 0 : !sir.u256
        \\    %word = sir.const 32 : !sir.u256
        \\    %buf = sir.malloc %word : !sir.u256 : !sir.ptr<1>
        \\    sir.cond_br %arg0 : !sir.u256, ^bb1, ^bb2
        \\  ^bb1:
        \\    sir.revert %buf : !sir.ptr<1>, %zero : !sir.u256
        \\  ^bb2:
        \\    sir.iret
        \\  ^bb3:
        \\    sir.iret
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);
    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));

    const module_before = try printModuleTextForTest(module);
    defer testing.allocator.free(module_before);
    const dot = try mlir_cfg.generateCFG(ctx, module, testing.allocator, .{ .mode = .sir });
    defer testing.allocator.free(dot);
    const module_after = try printModuleTextForTest(module);
    defer testing.allocator.free(module_after);

    try testing.expectEqualStrings(module_before, module_after);
    const graph = try parseSirDotGraph(testing.allocator, dot);
    defer graph.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 4), graph.nodes.len);
    try testing.expectEqual(@as(usize, 2), graph.edges.len);
    try testing.expectEqual(@as(usize, 1), graph.countEntryNodes());
    try testing.expectEqual(@as(usize, 1), graph.countUnreachableNodes());
    try testing.expectEqual(@as(usize, 1), graph.countRevertNodes());
    try testing.expectEqual(@as(usize, 1), graph.countTerm("sir.revert"));
    try testing.expectEqual(@as(usize, 1), graph.countTerm("sir.cond_br"));
    try graph.expectAllEdgesReferToKnownNodes();
    try graph.expectEveryCondBrHasTrueFalseEdges();
    try graph.expectRevertNodesHaveNoSuccessors();
}

test "compiler generates SIR CFG optimization diff without mutating module" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @const_false() {
        \\    %c0 = sir.const 0 : !sir.u256
        \\    sir.cond_br %c0 : !sir.u256, ^bb1, ^bb2
        \\  ^bb1:
        \\    sir.invalid
        \\  ^bb2:
        \\    sir.iret
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);
    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));

    const module_before = try printModuleTextForTest(module);
    defer testing.allocator.free(module_before);
    const diff = try mlir_cfg.generateSirOptimizationDiff(ctx, module, testing.allocator, false);
    defer diff.deinit(testing.allocator);
    const module_after = try printModuleTextForTest(module);
    defer testing.allocator.free(module_after);

    try testing.expectEqualStrings(module_before, module_after);
    const before_graph = try parseSirDotGraph(testing.allocator, diff.before);
    defer before_graph.deinit(testing.allocator);
    const after_graph = try parseSirDotGraph(testing.allocator, diff.after);
    defer after_graph.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), before_graph.nodes.len);
    try testing.expectEqual(@as(usize, 2), before_graph.edges.len);
    try testing.expectEqual(@as(usize, 1), before_graph.countTerm("sir.cond_br"));
    try testing.expectEqual(@as(usize, 1), before_graph.countTerm("sir.invalid"));
    try before_graph.expectEveryCondBrHasTrueFalseEdges();
    try testing.expect(after_graph.nodes.len < before_graph.nodes.len);
    try testing.expect(after_graph.edges.len < before_graph.edges.len);
    try testing.expectEqual(@as(usize, 0), after_graph.countTerm("sir.cond_br"));
    try testing.expectEqual(@as(usize, 0), after_graph.countTerm("sir.invalid"));
    try testing.expectEqual(@as(usize, 1), after_graph.countTerm("sir.iret"));
    try before_graph.expectAllEdgesReferToKnownNodes();
    try after_graph.expectAllEdgesReferToKnownNodes();
}

test "compiler marks proven refinement guards in Ora CFG overlay" {
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
    try testing.expect(hir_result.diagnostics.isEmpty());

    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    var vr = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer vr.deinit();

    try testing.expect(vr.success);
    try testing.expect(vr.proven_guard_ids.count() > 0);
    const module_before = try printModuleTextForTest(hir_result.module.raw_module);
    defer testing.allocator.free(module_before);

    const dot = try mlir_cfg.generateCFG(hir_result.context, hir_result.module.raw_module, testing.allocator, .{
        .mode = .ora,
        .proven_guard_ids = &vr.proven_guard_ids,
    });
    defer testing.allocator.free(dot);
    const module_after = try printModuleTextForTest(hir_result.module.raw_module);
    defer testing.allocator.free(module_after);

    try testing.expectEqualStrings(module_before, module_after);
    try testing.expect(std.mem.containsAtLeast(u8, dot, 1, "digraph \"ora_structured_cfg\""));
    try testing.expect(std.mem.containsAtLeast(u8, dot, 1, "proven_guard_count="));
    try testing.expect(std.mem.containsAtLeast(u8, dot, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, dot, 1, "proof=\"proven-erased\""));
    try testing.expectEqual(vr.proven_guard_ids.count(), std.mem.count(u8, dot, "proof=\"proven-erased\""));
}

test "compiler Ora CFG overlay distinguishes runtime refinement guards" {
    const runtime_source =
        \\pub fn runtime_guard(amount: NonZero<u256>) -> u256 {
        \\    return amount;
        \\}
    ;

    var runtime_compilation = try compileText(runtime_source);
    defer runtime_compilation.deinit();
    const runtime_hir = try runtime_compilation.db.lowerToHir(runtime_compilation.root_module_id);
    try testing.expect(runtime_hir.diagnostics.isEmpty());

    var runtime_verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer runtime_verifier.deinit();
    var runtime_vr = try runtime_verifier.runVerificationPass(runtime_hir.module.raw_module);
    defer runtime_vr.deinit();
    try testing.expect(runtime_vr.success);
    try testing.expectEqual(@as(usize, 0), runtime_vr.proven_guard_ids.count());

    const runtime_dot = try mlir_cfg.generateCFG(runtime_hir.context, runtime_hir.module.raw_module, testing.allocator, .{
        .mode = .ora,
        .proven_guard_ids = &runtime_vr.proven_guard_ids,
    });
    defer testing.allocator.free(runtime_dot);
    try testing.expect(std.mem.containsAtLeast(u8, runtime_dot, 1, "ora.refinement_guard"));
    try testing.expect(std.mem.containsAtLeast(u8, runtime_dot, 1, "proof=\"runtime\""));
    try testing.expect(!std.mem.containsAtLeast(u8, runtime_dot, 1, "proof=\"proven-erased\""));
}

// Runtime abiDecode structural helpers pin the guard shape for each decoded
// category: static head-only values, dynamic byte-padded values (string/bytes
// and the current mixed tuple), and dynamic word-only arrays such as slice[u256].
fn expectStaticAbiDecodeGuardBeforePayloadLoad(fn_text: []const u8) !void {
    const branch_index = std.mem.indexOf(u8, fn_text, " ? @") orelse return error.TestUnexpectedResult;
    const before_guard = fn_text[0..branch_index];
    const after_guard = fn_text[branch_index..];
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, before_guard, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 1, "mload256"));
}

fn expectDynamicAbiDecodeGuardChain(fn_text: []const u8) !void {
    const branch_index = std.mem.indexOf(u8, fn_text, " ? @") orelse return error.TestUnexpectedResult;
    const before_guard = fn_text[0..branch_index];
    const after_guard = fn_text[branch_index..];
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, before_guard, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 2, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 1, "mload8"));
}

fn expectDynamicAbiDecodeWordGuardChain(fn_text: []const u8) !void {
    const branch_index = std.mem.indexOf(u8, fn_text, " ? @") orelse return error.TestUnexpectedResult;
    const before_guard = fn_text[0..branch_index];
    const after_guard = fn_text[branch_index..];
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, before_guard, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 2, "mload256"));
}

fn expectMixedDynamicTupleCarrierShape(fn_text: []const u8) !void {
    // The dedicated mixed dynamic tuple branch allocates a 2-slot tuple carrier,
    // stores the static u256, then stores the string/bytes tail pointer.
    try expectOrderedNeedles(fn_text, &.{ "const 0x40", "mload256", "malloc", "mstore256", "add", "mstore256" });
    try testing.expect(std.mem.containsAtLeast(u8, fn_text, 1, "const 0x20"));
    try testing.expect(std.mem.containsAtLeast(u8, fn_text, 2, "malloc"));
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

test "OraToSIR folds redundant integer/u256 bitcast round trips" {
    const source_text =
        \\contract Counter {
        \\    storage var counter: u256;
        \\
        \\    pub fn increment() {
        \\        counter = counter + 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const increment_fn = try functionSlice(rendered, "increment");

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, increment_fn, "sload"));
    try testing.expect(!std.mem.containsAtLeast(u8, increment_fn, 1, "bitcast"));
}

test "Phase0 bitcast folder runs when final manual bitcast fold is skipped" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @same_width_roundtrip(%arg0: !sir.u256) {
        \\    %0 = sir.bitcast %arg0 : !sir.u256 : i256
        \\    %1 = sir.bitcast %0 : i256 : !sir.u256
        \\    sir.iret %1
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);
    setModuleBoolAttr(ctx, module, "ora.phase0.skip_manual_bitcast_fold");

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.bitcast"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.iret %arg0"));
}

test "Phase0 framework canonicalizer folds identity sir.bitcast without manual fold" {
    inline for (.{ false, true }) |debug_info| {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);

        const text =
            \\module {
            \\  func.func @identity(%arg0: !sir.u256) {
            \\    %0 = sir.bitcast %arg0 : !sir.u256 : !sir.u256
            \\    sir.iret %0
            \\  }
            \\}
        ;
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);
        setModuleBoolAttr(ctx, module, "ora.phase0.skip_manual_bitcast_fold");
        setModuleBoolAttr(ctx, module, "ora.phase0.run_sir_framework_canonicalizer");

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, debug_info));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.bitcast"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.iret %arg0"));
    }
}

test "Phase0 framework canonicalizer folds same-width sir.bitcast round trip" {
    inline for (.{ false, true }) |debug_info| {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);

        const text =
            \\module {
            \\  func.func @same_width_roundtrip(%arg0: !sir.u256) {
            \\    %0 = sir.bitcast %arg0 : !sir.u256 : i256
            \\    %1 = sir.bitcast %0 : i256 : !sir.u256
            \\    sir.iret %1
            \\  }
            \\}
        ;
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);
        setModuleBoolAttr(ctx, module, "ora.phase0.skip_manual_bitcast_fold");
        setModuleBoolAttr(ctx, module, "ora.phase0.run_sir_framework_canonicalizer");

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, debug_info));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.bitcast"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.iret %arg0"));
    }
}

test "Phase0 framework canonicalizer removes dead pure SIR values" {
    inline for (.{ false, true }) |debug_info| {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);

        const text =
            \\module {
            \\  func.func @dead_pure(%arg0: !sir.u256) {
            \\    %dead = sir.add %arg0 : !sir.u256, %arg0 : !sir.u256 : !sir.u256
            \\    sir.iret %arg0
            \\  }
            \\}
        ;
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);
        setModuleBoolAttr(ctx, module, "ora.phase0.skip_manual_bitcast_fold");
        setModuleBoolAttr(ctx, module, "ora.phase0.run_sir_framework_canonicalizer");

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, debug_info));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.add"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.iret %arg0"));
    }
}

test "Phase0 framework canonicalizer folds sir.eq and sir.iszero" {
    inline for (.{ false, true }) |debug_info| {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);

        const text =
            \\module {
            \\  func.func @eq_same(%arg0: !sir.u256) {
            \\    %0 = sir.eq %arg0 : !sir.u256, %arg0 : !sir.u256 : !sir.u256
            \\    sir.iret %0
            \\  }
            \\  func.func @eq_const() {
            \\    %c7 = sir.const 7 : !sir.u256
            \\    %c8 = sir.const 8 : !sir.u256
            \\    %0 = sir.eq %c7 : !sir.u256, %c8 : !sir.u256 : !sir.u256
            \\    sir.iret %0
            \\  }
            \\  func.func @iszero_const() {
            \\    %c0 = sir.const 0 : !sir.u256
            \\    %0 = sir.iszero %c0 : !sir.u256 : !sir.u256
            \\    sir.iret %0
            \\  }
            \\  func.func @wide_const_eq() {
            \\    %max = sir.const 115792089237316195423570985008687907853269984665640564039457584007913129639935 : !sir.u256
            \\    %0 = sir.eq %max : !sir.u256, %max : !sir.u256 : !sir.u256
            \\    sir.iret %0
            \\  }
            \\  func.func @wide_const_iszero() {
            \\    %max = sir.const 115792089237316195423570985008687907853269984665640564039457584007913129639935 : !sir.u256
            \\    %0 = sir.iszero %max : !sir.u256 : !sir.u256
            \\    sir.iret %0
            \\  }
            \\  func.func @add_mul_const() {
            \\    %c2 = sir.const 2 : !sir.u256
            \\    %c3 = sir.const 3 : !sir.u256
            \\    %sum = sir.add %c2 : !sir.u256, %c3 : !sir.u256 : !sir.u256
            \\    %product = sir.mul %c2 : !sir.u256, %c3 : !sir.u256 : !sir.u256
            \\    sir.iret %sum, %product
            \\  }
            \\  func.func @more_const_folds() {
            \\    %c0 = sir.const 0 : !sir.u256
            \\    %c1 = sir.const 1 : !sir.u256
            \\    %c2 = sir.const 2 : !sir.u256
            \\    %c8 = sir.const 8 : !sir.u256
            \\    %c31 = sir.const 31 : !sir.u256
            \\    %c32 = sir.const 32 : !sir.u256
            \\    %c255 = sir.const 255 : !sir.u256
            \\    %c256 = sir.const 256 : !sir.u256
            \\    %max = sir.const 115792089237316195423570985008687907853269984665640564039457584007913129639935 : !sir.u256
            \\    %sub = sir.sub %c2 : !sir.u256, %c1 : !sir.u256 : !sir.u256
            \\    %and = sir.and %max : !sir.u256, %c1 : !sir.u256 : !sir.u256
            \\    %or = sir.or %c1 : !sir.u256, %c2 : !sir.u256 : !sir.u256
            \\    %xor = sir.xor %c1 : !sir.u256, %c2 : !sir.u256 : !sir.u256
            \\    %not = sir.not %c0 : !sir.u256 : !sir.u256
            \\    %lt = sir.lt %c1 : !sir.u256, %c2 : !sir.u256 : !sir.u256
            \\    %gt = sir.gt %c1 : !sir.u256, %c2 : !sir.u256 : !sir.u256
            \\    %slt = sir.slt %max : !sir.u256, %c1 : !sir.u256 : !sir.u256
            \\    %sgt = sir.sgt %max : !sir.u256, %c1 : !sir.u256 : !sir.u256
            \\    %byte = sir.byte %c31 : !sir.u256, %max : !sir.u256 : !sir.u256
            \\    %byte_oob = sir.byte %c32 : !sir.u256, %max : !sir.u256 : !sir.u256
            \\    %shl = sir.shl %c8 : !sir.u256, %c1 : !sir.u256 : !sir.u256
            \\    %shr = sir.shr %c8 : !sir.u256, %max : !sir.u256 : !sir.u256
            \\    %shl_oob = sir.shl %c256 : !sir.u256, %c1 : !sir.u256 : !sir.u256
            \\    %shr_oob = sir.shr %c256 : !sir.u256, %max : !sir.u256 : !sir.u256
            \\    %sar_neg = sir.sar %c255 : !sir.u256, %max : !sir.u256 : !sir.u256
            \\    %sar_oob_neg = sir.sar %c256 : !sir.u256, %max : !sir.u256 : !sir.u256
            \\    %sar_oob_pos = sir.sar %c256 : !sir.u256, %c1 : !sir.u256 : !sir.u256
            \\    sir.iret %sub, %and, %or, %xor, %not, %lt, %gt, %slt, %sgt, %byte, %byte_oob, %shl, %shr, %shl_oob, %shr_oob, %sar_neg, %sar_oob_neg, %sar_oob_pos
            \\  }
            \\}
        ;
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);
        setModuleBoolAttr(ctx, module, "ora.phase0.skip_manual_bitcast_fold");
        setModuleBoolAttr(ctx, module, "ora.phase0.run_sir_framework_canonicalizer");

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, debug_info));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.eq"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.iszero"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.add"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.mul"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.sub"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.and"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.or"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.xor"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.not"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.byte"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.shl"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.shr"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.sar"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.lt"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.gt"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.slt"));
        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.sgt"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 3, "sir.const 1 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "sir.const 0 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 3 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 5 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 6 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 255 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 256 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 452312848583266388373324160190187140051835877600158453279131187530910662655 : !sir.u256"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 115792089237316195423570985008687907853269984665640564039457584007913129639935 : !sir.u256"));
    }
}

test "Phase0 framework canonicalizer folds SIR branch peepholes" {
    inline for (.{ false, true }) |debug_info| {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);

        const text =
            \\module {
            \\  func.func @same_dest(%arg0: !sir.u256) {
            \\    sir.cond_br %arg0 : !sir.u256, ^bb1, ^bb1
            \\  ^bb1:
            \\    sir.iret
            \\  }
            \\  func.func @const_false() {
            \\    %c0 = sir.const 0 : !sir.u256
            \\    sir.cond_br %c0 : !sir.u256, ^bb1, ^bb2
            \\  ^bb1:
            \\    sir.invalid
            \\  ^bb2:
            \\    sir.iret
            \\  }
            \\  func.func @double_iszero(%arg0: !sir.u256) {
            \\    %0 = sir.iszero %arg0 : !sir.u256 : !sir.u256
            \\    %1 = sir.iszero %0 : !sir.u256 : !sir.u256
            \\    sir.cond_br %1 : !sir.u256, ^bb1, ^bb2
            \\  ^bb1:
            \\    sir.iret
            \\  ^bb2:
            \\    sir.invalid
            \\  }
            \\  func.func @br_to_br(%arg0: !sir.u256) {
            \\    sir.br ^bb1(%arg0 : !sir.u256)
            \\  ^bb1(%x: !sir.u256):
            \\    sir.br ^bb2(%x : !sir.u256)
            \\  ^bb2(%y: !sir.u256):
            \\    sir.iret %y
            \\  }
            \\}
        ;
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);
        setModuleBoolAttr(ctx, module, "ora.phase0.skip_manual_bitcast_fold");
        setModuleBoolAttr(ctx, module, "ora.phase0.run_sir_framework_canonicalizer");

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, debug_info));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        const same_dest = try oraFunctionSlice(rendered, "same_dest");
        try testing.expect(!std.mem.containsAtLeast(u8, same_dest, 1, "sir.cond_br"));
        try testing.expect(std.mem.containsAtLeast(u8, same_dest, 1, "sir.br ^bb1"));

        const const_false = try oraFunctionSlice(rendered, "const_false");
        try testing.expect(!std.mem.containsAtLeast(u8, const_false, 1, "sir.cond_br"));
        try testing.expect(!std.mem.containsAtLeast(u8, const_false, 1, "sir.invalid"));
        try testing.expect(std.mem.containsAtLeast(u8, const_false, 1, "sir.iret"));

        const double_iszero = try oraFunctionSlice(rendered, "double_iszero");
        try testing.expect(!std.mem.containsAtLeast(u8, double_iszero, 1, "sir.iszero"));
        try testing.expect(std.mem.containsAtLeast(u8, double_iszero, 1, "sir.cond_br %arg0"));

        const br_to_br = try oraFunctionSlice(rendered, "br_to_br");
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, br_to_br, "sir.br"));
        try testing.expect(std.mem.containsAtLeast(u8, br_to_br, 1, "sir.iret"));
    }
}

test "SIR framework canonicalizer runs by default" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @same_dest(%arg0: !sir.u256) {
        \\    sir.cond_br %arg0 : !sir.u256, ^bb1, ^bb1
        \\  ^bb1:
        \\    sir.iret
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

    const same_dest = try oraFunctionSlice(rendered, "same_dest");
    try testing.expect(!std.mem.containsAtLeast(u8, same_dest, 1, "sir.cond_br"));
    try testing.expect(std.mem.containsAtLeast(u8, same_dest, 1, "sir.br ^bb1"));
}

test "Ora canonicalization runs framework SymbolDCE with Ora root visibility" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @public_entry() attributes {ora.visibility = "pub"} {
        \\      func.call @live_private() : () -> ()
        \\      return
        \\    }
        \\    func.func @init() attributes {ora.init = true, ora.visibility = "private"} {
        \\      return
        \\    }
        \\    func.func @debug_probe() attributes {ora.debug_root = true, ora.visibility = "private"} {
        \\      func.call @debug_live() : () -> ()
        \\      return
        \\    }
        \\    func.func @dispatcher_root() attributes {ora.symbol_root = true, ora.visibility = "private"} {
        \\      func.call @dispatcher_live() : () -> ()
        \\      return
        \\    }
        \\    func.func @plain_unannotated() {
        \\      return
        \\    }
        \\    func.func @live_private() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\    func.func @debug_live() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\    func.func @dispatcher_live() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\    func.func @dead_private() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));

    var stats: mlir.OraMlirPassStatisticsC = std.mem.zeroes(mlir.OraMlirPassStatisticsC);
    try testing.expect(mlir.oraCanonicalizeOraMLIRWithStatisticsOut(ctx, module, false, &stats));
    try testing.expectEqual(@as(u64, 1), stats.ora_symbols_dced);

    const rendered = try printModuleTextForTest(module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @public_entry"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @init"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @debug_probe"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @dispatcher_root"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @plain_unannotated"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @live_private"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @debug_live"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @dispatcher_live"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "func.func @dead_private"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sym_visibility"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.symbol_dce.temp_visibility"));
}

test "Ora SymbolDCE removes private call islands unreachable from roots" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @entry() attributes {ora.visibility = "pub"} {
        \\      func.call @live_a() : () -> ()
        \\      return
        \\    }
        \\    func.func @live_a() attributes {ora.visibility = "private"} {
        \\      func.call @live_b() : () -> ()
        \\      return
        \\    }
        \\    func.func @live_b() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\    func.func @dead_a() attributes {ora.visibility = "private"} {
        \\      func.call @dead_b() : () -> ()
        \\      return
        \\    }
        \\    func.func @dead_b() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));

    var stats: mlir.OraMlirPassStatisticsC = std.mem.zeroes(mlir.OraMlirPassStatisticsC);
    try testing.expect(mlir.oraCanonicalizeOraMLIRWithStatisticsOut(ctx, module, false, &stats));
    try testing.expectEqual(@as(u64, 2), stats.ora_symbols_dced);

    const rendered = try printModuleTextForTest(module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @entry"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @live_a"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @live_b"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "func.func @dead_a"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "func.func @dead_b"));
}

test "Ora SymbolDCE preserves dotted imported private helpers reachable from roots" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @entry() attributes {ora.visibility = "pub"} {
        \\      func.call @dep.required() : () -> ()
        \\      return
        \\    }
        \\    func.func @dep.required() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\    func.func @dep.dead() attributes {ora.visibility = "private"} {
        \\      return
        \\    }
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));

    var stats: mlir.OraMlirPassStatisticsC = std.mem.zeroes(mlir.OraMlirPassStatisticsC);
    try testing.expect(mlir.oraCanonicalizeOraMLIRWithStatisticsOut(ctx, module, false, &stats));
    try testing.expectEqual(@as(u64, 1), stats.ora_symbols_dced);

    const rendered = try printModuleTextForTest(module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @entry"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @dep.required"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "func.func @dep.dead"));
}

test "compiler source MLIR SymbolDCE keeps public root and prunes generated private helpers" {
    const source_text =
        \\contract SymbolDceSource {
        \\    pub fn entry() -> u256 {
        \\        return live();
        \\    }
        \\
        \\    fn live() -> u256 {
        \\        return 7;
        \\    }
        \\
        \\    fn dead() -> u256 {
        \\        return 9;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    var stats: mlir.OraMlirPassStatisticsC = std.mem.zeroes(mlir.OraMlirPassStatisticsC);
    try testing.expect(mlir.oraCanonicalizeOraMLIRWithStatisticsOut(hir_result.context, hir_result.module.raw_module, false, &stats));
    try testing.expectEqual(@as(u64, 2), stats.ora_symbols_dced);

    const rendered = try printModuleTextForTest(hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @entry"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "func.func @live"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "func.func @dead"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.selector"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.visibility = \"pub\""));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sym_visibility"));
}

test "compiler source SymbolDCE preserves dispatcher public roots" {
    const source_text =
        \\contract DispatcherRoots {
        \\    pub fn first() -> u256 {
        \\        return 1;
        \\    }
        \\
        \\    pub fn second(value: u256) -> u256 {
        \\        return value;
        \\    }
        \\
        \\    fn dead() -> u256 {
        \\        return 9;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    var stats: mlir.OraMlirPassStatisticsC = std.mem.zeroes(mlir.OraMlirPassStatisticsC);
    try testing.expect(mlir.oraCanonicalizeOraMLIRWithStatisticsOut(hir_result.context, hir_result.module.raw_module, false, &stats));
    try testing.expectEqual(@as(u64, 1), stats.ora_symbols_dced);

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn first:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn second:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "fn dead:"));
}

test "OraToSIR lowers exact no-result switch statements to structured sir.switch" {
    const source_text =
        \\pub fn classify(tag: u256) -> u256 {
        \\    switch (tag) {
        \\        0 => { return 10; }
        \\        1 => { return 20; }
        \\        2 => { return 30; }
        \\        else => { return 99; }
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const classify_fn = try functionSlice(rendered, "classify");

    try testing.expect(std.mem.containsAtLeast(u8, classify_fn, 1, "switch "));
    try testing.expect(std.mem.containsAtLeast(u8, classify_fn, 1, "0x0 =>"));
    try testing.expect(std.mem.containsAtLeast(u8, classify_fn, 1, "0x1 =>"));
    try testing.expect(std.mem.containsAtLeast(u8, classify_fn, 1, "0x2 =>"));
    try testing.expect(std.mem.containsAtLeast(u8, classify_fn, 1, "default =>"));
    try testing.expect(!std.mem.containsAtLeast(u8, classify_fn, 1, " = eq "));
}

test "OraToSIR keeps range switch statements on condition-chain fallback" {
    const source_text =
        \\pub fn classify_range(tag: u256) -> u256 {
        \\    switch (tag) {
        \\        0...9 => { return 1; }
        \\        else => { return 2; }
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const classify_fn = try functionSlice(rendered, "classify_range");

    try testing.expect(!std.mem.containsAtLeast(u8, classify_fn, 1, "switch "));
    try testing.expect(std.mem.containsAtLeast(u8, classify_fn, 1, " = lt "));
    try testing.expect(std.mem.containsAtLeast(u8, classify_fn, 1, " = gt "));
}

test "Phase3 release SIR optimization materializes framework folds" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @release_framework(%arg0: !sir.u256) {
        \\    %c2 = sir.const 2 : !sir.u256
        \\    %c3 = sir.const 3 : !sir.u256
        \\    %max = sir.const 115792089237316195423570985008687907853269984665640564039457584007913129639935 : !sir.u256
        \\    %sum = sir.add %c2 : !sir.u256, %c3 : !sir.u256 : !sir.u256
        \\    %diff = sir.sub %c3 : !sir.u256, %c2 : !sir.u256 : !sir.u256
        \\    %product = sir.mul %c2 : !sir.u256, %c3 : !sir.u256 : !sir.u256
        \\    %lt = sir.lt %c2 : !sir.u256, %c3 : !sir.u256 : !sir.u256
        \\    %gt = sir.gt %c3 : !sir.u256, %c2 : !sir.u256 : !sir.u256
        \\    %slt = sir.slt %max : !sir.u256, %c2 : !sir.u256 : !sir.u256
        \\    %sgt = sir.sgt %c2 : !sir.u256, %max : !sir.u256 : !sir.u256
        \\    %and = sir.and %c3 : !sir.u256, %c2 : !sir.u256 : !sir.u256
        \\    %or = sir.or %c2 : !sir.u256, %c3 : !sir.u256 : !sir.u256
        \\    %xor = sir.xor %c2 : !sir.u256, %c3 : !sir.u256 : !sir.u256
        \\    %same = sir.eq %arg0 : !sir.u256, %arg0 : !sir.u256 : !sir.u256
        \\    sir.iret %sum, %diff, %product, %lt, %gt, %slt, %sgt, %and, %or, %xor, %same
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

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.add"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.sub"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.mul"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.lt"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.slt"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.sgt"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.and"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.or"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.xor"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.eq"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 1 : !sir.u256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 2 : !sir.u256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 3 : !sir.u256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 5 : !sir.u256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.const 6 : !sir.u256"));
}

test "OraToSIR keeps width-changing bitcast round trips" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @guard(%arg0: !sir.u256) {
        \\    %0 = sir.bitcast %arg0 : !sir.u256 : i1
        \\    %1 = sir.bitcast %0 : i1 : !sir.u256
        \\    sir.iret %1
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

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "sir.bitcast"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ": !sir.u256 : i1"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ": i1 : !sir.u256"));
}

test "Phase0 framework canonicalizer keeps width-changing sir.bitcast round trip" {
    inline for (.{ false, true }) |debug_info| {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);

        const text =
            \\module {
            \\  func.func @guard(%arg0: !sir.u256) {
            \\    %0 = sir.bitcast %arg0 : !sir.u256 : i1
            \\    %1 = sir.bitcast %0 : i1 : !sir.u256
            \\    sir.iret %1
            \\  }
            \\}
        ;
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);
        setModuleBoolAttr(ctx, module, "ora.phase0.run_sir_framework_canonicalizer");

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, debug_info));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "sir.bitcast"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ": !sir.u256 : i1"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ": i1 : !sir.u256"));
    }
}

test "Phase2 retained carrier safety net folds normalized narrow round trip only" {
    const text =
        \\module {
        \\  func.func @masked_carrier(%arg0: !sir.u256) {
        \\    %mask = sir.const 255 : !sir.u256
        \\    %masked = sir.and %arg0 : !sir.u256, %mask : !sir.u256 : !sir.u256
        \\    %narrow = sir.bitcast %masked : !sir.u256 : i8
        \\    %carrier = sir.bitcast %narrow : i8 : !sir.u256
        \\    sir.iret %carrier
        \\  }
        \\}
    ;

    {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.bitcast"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.and"));
    }

    {
        const ctx = createOraMlirContext();
        defer mlir.oraContextDestroy(ctx);
        const module = try parseOraModule(ctx, text);
        defer mlir.oraModuleDestroy(module);
        setModuleBoolAttr(ctx, module, "ora.phase0.skip_manual_bitcast_fold");
        setModuleBoolAttr(ctx, module, "ora.phase0.run_sir_framework_canonicalizer");

        try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
        try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "sir.bitcast"));
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.and"));
    }
}

test "Phase2 address refinement guard lowers without i160 carrier round trip" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\pub fn keep(addr: NonZeroAddress) -> bool
        \\    requires(addr != std.constants.ZERO_ADDRESS)
        \\{
        \\    return addr != std.constants.ZERO_ADDRESS;
        \\}
    ;

    const rendered = try renderSirTextForSourceWithAttrs(source_text, false, false, false);
    defer testing.allocator.free(rendered);

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, ": !sir.u256 : i160"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, ": i160 : !sir.u256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, " and "));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "revert"));
}

test "Phase0 framework canonicalizer converts loop-containing source in release and debug modes" {
    const source_text =
        \\pub fn loop_sum(limit: u256) -> u256 {
        \\    var value: u256 = 0;
        \\    var total: u256 = 0;
        \\    while (value < limit) {
        \\        total = total + value;
        \\        value = value + 1;
        \\    }
        \\    return total;
        \\}
    ;

    inline for (.{ false, true }) |debug_info| {
        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());

        setModuleBoolAttr(hir_result.context, hir_result.module.raw_module, "ora.phase0.skip_manual_bitcast_fold");
        setModuleBoolAttr(hir_result.context, hir_result.module.raw_module, "ora.phase0.run_sir_framework_canonicalizer");
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, debug_info));

        const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
        defer testing.allocator.free(rendered);
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "loop_sum"));
    }
}

test "Phase0 framework canonicalizer characterizes source bitcast residuals" {
    const Case = struct {
        name: []const u8,
        source: []const u8,
    };

    const cases = [_]Case{
        .{
            .name = "storage bitcast-heavy",
            .source =
            \\contract Counter {
            \\    storage var counter: u256;
            \\
            \\    pub fn increment() {
            \\        counter = counter + 1;
            \\    }
            \\}
            ,
        },
        .{
            .name = "signed narrow",
            .source =
            \\pub fn signed_div(a: i8, b: i8) -> i8 {
            \\    return a / b;
            \\}
            ,
        },
        .{
            .name = "loop-containing",
            .source =
            \\pub fn loop_sum(limit: u256) -> u256 {
            \\    var value: u256 = 0;
            \\    var total: u256 = 0;
            \\    while (value < limit) {
            \\        total = total + value;
            \\        value = value + 1;
            \\    }
            \\    return total;
            \\}
            ,
        },
    };

    for (cases) |case| {
        const current_release = try countSirBitcastsForSource(case.source, false, false, false);
        const manual_disabled_release = try countSirBitcastsForSource(case.source, false, true, false);
        const framework_release = try countSirBitcastsForSource(case.source, false, true, true);
        const framework_debug = try countSirBitcastsForSource(case.source, true, true, true);

        try testing.expectEqual(current_release, manual_disabled_release);
        try testing.expectEqual(manual_disabled_release, framework_release);
        try testing.expectEqual(framework_release, framework_debug);
        try testing.expectEqual(@as(usize, 0), framework_release);
    }
}

test "Phase1 Ora canonicalizer folds refinement bridge round trips" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @base_roundtrip(%arg0: !ora.int<256, false>) -> !ora.int<256, false> {
        \\    %0 = ora.base_to_refinement %arg0 : !ora.int<256, false> -> !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>
        \\    %1 = ora.refinement_to_base %0 : !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1> -> !ora.int<256, false>
        \\    return %1 : !ora.int<256, false>
        \\  }
        \\  func.func @refinement_roundtrip(%arg0: !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>) -> !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1> {
        \\    %0 = ora.refinement_to_base %arg0 : !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1> -> !ora.int<256, false>
        \\    %1 = ora.base_to_refinement %0 : !ora.int<256, false> -> !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>
        \\    return %1 : !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.base_to_refinement"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.refinement_to_base"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "return %arg0 : !ora.int<256, false>"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "return %arg0 : !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>"));
}

test "Phase1 Ora canonicalizer folds address carrier round trips" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @i160_roundtrip(%arg0: i160) -> i160 {
        \\    %0 = ora.i160.to.addr %arg0 : i160 -> !ora.address
        \\    %1 = ora.addr.to.i160 %0 : !ora.address -> i160
        \\    return %1 : i160
        \\  }
        \\  func.func @address_roundtrip(%arg0: !ora.address) -> !ora.address {
        \\    %0 = ora.addr.to.i160 %arg0 : !ora.address -> i160
        \\    %1 = ora.i160.to.addr %0 : i160 -> !ora.address
        \\    return %1 : !ora.address
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.addr.to.i160"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.i160.to.addr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "return %arg0 : i160"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "return %arg0 : !ora.address"));
}

test "Phase1 Ora canonicalizer folds safe arithmetic identities only" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @safe_identities(%arg0: i256) -> (i256, i256, i256, i256, i256, i256, i256, i256, i256) {
        \\    %zero = arith.constant 0 : i256
        \\    %one = arith.constant 1 : i256
        \\    %three = arith.constant 3 : i256
        \\    %seven = arith.constant 7 : i256
        \\    %add_rhs = arith.addi %arg0, %zero : i256
        \\    %add_lhs = arith.addi %zero, %arg0 : i256
        \\    %sub_rhs = arith.subi %arg0, %zero : i256
        \\    %mul_rhs = arith.muli %arg0, %one : i256
        \\    %mul_lhs = arith.muli %one, %arg0 : i256
        \\    %mul_zero = arith.muli %arg0, %zero : i256
        \\    %div_one = arith.divui %arg0, %one : i256
        \\    %rem_const = arith.remui %seven, %three : i256
        \\    %rem_one = arith.remui %arg0, %one : i256
        \\    return %add_rhs, %add_lhs, %sub_rhs, %mul_rhs, %mul_lhs, %mul_zero, %div_one, %rem_const, %rem_one : i256, i256, i256, i256, i256, i256, i256, i256, i256
        \\  }
        \\  func.func @kept_non_identities(%arg0: i256) -> (i256, i256, i256) {
        \\    %zero = arith.constant 0 : i256
        \\    %sub = arith.subi %zero, %arg0 : i256
        \\    %div = arith.divui %zero, %arg0 : i256
        \\    %rem = arith.remui %arg0, %zero : i256
        \\    return %sub, %div, %rem : i256, i256, i256
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, rendered, "arith.addi"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "arith.subi"));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, rendered, "arith.muli"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "arith.divui"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "arith.remui"));
}

test "Phase1 Ora canonicalizer folds wrapping arithmetic" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @wrapping_identities(%arg0: !ora.int<8, false>) -> (!ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>) {
        \\    %zero = ora.const "zero" : !ora.int<8, false> = 0 : i8
        \\    %one = ora.const "one" : !ora.int<8, false> = 1 : i8
        \\    %add_rhs = ora.add_wrapping %arg0, %zero : !ora.int<8, false>, !ora.int<8, false> -> !ora.int<8, false>
        \\    %add_lhs = ora.add_wrapping %zero, %arg0 : !ora.int<8, false>, !ora.int<8, false> -> !ora.int<8, false>
        \\    %sub_rhs = ora.sub_wrapping %arg0, %zero : !ora.int<8, false>, !ora.int<8, false> -> !ora.int<8, false>
        \\    %mul_rhs = ora.mul_wrapping %arg0, %one : !ora.int<8, false>, !ora.int<8, false> -> !ora.int<8, false>
        \\    %mul_lhs = ora.mul_wrapping %one, %arg0 : !ora.int<8, false>, !ora.int<8, false> -> !ora.int<8, false>
        \\    %shl_zero = ora.shl_wrapping %arg0, %zero : !ora.int<8, false>, !ora.int<8, false> -> !ora.int<8, false>
        \\    %shr_zero = ora.shr_wrapping %arg0, %zero : !ora.int<8, false>, !ora.int<8, false> -> !ora.int<8, false>
        \\    return %add_rhs, %add_lhs, %sub_rhs, %mul_rhs, %mul_lhs, %shl_zero, %shr_zero : !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>, !ora.int<8, false>
        \\  }
        \\  func.func @wrapping_constants() -> (i8, i8, i8, i8, i8) {
        \\    %zero = arith.constant 0 : i8
        \\    %one = arith.constant 1 : i8
        \\    %two = arith.constant 2 : i8
        \\    %sixteen = arith.constant 16 : i8
        \\    %all_ones = arith.constant -1 : i8
        \\    %add = ora.add_wrapping %all_ones, %one : i8, i8 -> i8
        \\    %sub = ora.sub_wrapping %zero, %one : i8, i8 -> i8
        \\    %mul = ora.mul_wrapping %sixteen, %sixteen : i8, i8 -> i8
        \\    %shl = ora.shl_wrapping %one, %two : i8, i8 -> i8
        \\    %shr = ora.shr_wrapping %all_ones, %one : i8, i8 -> i8
        \\    return %add, %sub, %mul, %shl, %shr : i8, i8, i8, i8, i8
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const identities = try oraFunctionSlice(rendered, "wrapping_identities");
    try testing.expect(!std.mem.containsAtLeast(u8, identities, 1, "ora.add_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, identities, 1, "ora.sub_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, identities, 1, "ora.mul_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, identities, 1, "ora.shl_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, identities, 1, "ora.shr_wrapping"));

    const constants = try oraFunctionSlice(rendered, "wrapping_constants");
    try testing.expect(!std.mem.containsAtLeast(u8, constants, 1, "ora.add_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, constants, 1, "ora.sub_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, constants, 1, "ora.mul_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, constants, 1, "ora.shl_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, constants, 1, "ora.shr_wrapping"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "arith.constant 0 : i8"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "arith.constant -1 : i8"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "arith.constant 4 : i8"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "arith.constant 127 : i8"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "return %c0_i8, %c_neg1_i8, %c0_i8, %c4_i8, %c127_i8"));
}

test "Phase1 Ora canonicalizer folds bounded power" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @power_identities(%arg0: !ora.int<256, false>) -> (!ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>) {
        \\    %zero = ora.const "zero" : !ora.int<256, false> = 0 : i256
        \\    %one = ora.const "one" : !ora.int<256, false> = 1 : i256
        \\    %pow_zero = ora.power %arg0, %zero : !ora.int<256, false>, !ora.int<256, false> -> !ora.int<256, false>
        \\    %pow_one = ora.power %arg0, %one : !ora.int<256, false>, !ora.int<256, false> -> !ora.int<256, false>
        \\    %one_pow = ora.power %one, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> !ora.int<256, false>
        \\    return %pow_zero, %pow_one, %one_pow : !ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>
        \\  }
        \\  func.func @power_constants() -> (i8, i8, i8) {
        \\    %zero = arith.constant 0 : i8
        \\    %two = arith.constant 2 : i8
        \\    %three = arith.constant 3 : i8
        \\    %sixteen = arith.constant 16 : i8
        \\    %pow = ora.power %two, %three : i8, i8 -> i8
        \\    %wrap = ora.power %sixteen, %two : i8, i8 -> i8
        \\    %zero_zero = ora.power %zero, %zero : i8, i8 -> i8
        \\    return %pow, %wrap, %zero_zero : i8, i8, i8
        \\  }
        \\  func.func @power_large_exponent_kept() -> i16 {
        \\    %two = arith.constant 2 : i16
        \\    %big = arith.constant 2048 : i16
        \\    %pow = ora.power %two, %big : i16, i16 -> i16
        \\    return %pow : i16
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const identities = try oraFunctionSlice(rendered, "power_identities");
    try testing.expect(!std.mem.containsAtLeast(u8, identities, 1, "ora.power"));
    try testing.expect(std.mem.containsAtLeast(u8, identities, 1, "arith.constant 1 : !ora.int<256, false>"));
    try testing.expect(std.mem.containsAtLeast(u8, identities, 1, "return "));
    try testing.expect(std.mem.containsAtLeast(u8, identities, 1, "%arg0"));

    const constants = try oraFunctionSlice(rendered, "power_constants");
    try testing.expect(!std.mem.containsAtLeast(u8, constants, 1, "ora.power"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "arith.constant 8 : i8"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "arith.constant 0 : i8"));
    try testing.expect(std.mem.containsAtLeast(u8, constants, 1, "arith.constant 1 : i8"));

    const large_exponent = try oraFunctionSlice(rendered, "power_large_exponent_kept");
    try testing.expect(std.mem.containsAtLeast(u8, large_exponent, 1, "ora.power"));
}

test "Phase1 Ora canonicalizer folds safe comparisons" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @folded_same(%arg0: !ora.int<256, false>) -> (i1, i1, i1, i1) {
        \\    %eq = ora.cmp "eq", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    %ne = ora.cmp "ne", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    %le = ora.cmp "le", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    %slt = ora.cmp "slt", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    return %eq, %ne, %le, %slt : i1, i1, i1, i1
        \\  }
        \\  func.func @folded_constants() -> (i1, i1, i1, i1) {
        \\    %u2 = ora.const "u2" : !ora.int<256, false> = 2 : i256
        \\    %u3 = ora.const "u3" : !ora.int<256, false> = 3 : i256
        \\    %minus_one = arith.constant -1 : i8
        \\    %plus_one = arith.constant 1 : i8
        \\    %ult = ora.cmp "ult", %u2, %u3 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    %uge = ora.cmp "uge", %u2, %u3 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    %signed = ora.cmp "slt", %minus_one, %plus_one : i8, i8 -> i1
        \\    %unsigned = ora.cmp "ult", %minus_one, %plus_one : i8, i8 -> i1
        \\    return %ult, %uge, %signed, %unsigned : i1, i1, i1, i1
        \\  }
        \\  func.func @kept_dynamic(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) -> i1 {
        \\    %lt = ora.cmp "lt", %arg0, %arg1 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    return %lt : i1
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "ora.cmp "));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @kept_dynamic"));
}

test "Phase1 Ora canonicalizer preserves kept-check operands" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @kept_check_if(%arg0: i1) -> i256 {
        \\    %0 = scf.if %arg0 -> (i1) {
        \\      %true = arith.constant true
        \\      scf.yield %true : i1
        \\    } else {
        \\      %false = arith.constant false
        \\      scf.yield %false : i1
        \\    }
        \\    ora.ensures %0 : i1
        \\    %c0_i256 = arith.constant 0 : i256
        \\    return %c0_i256 : i256
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));
    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.ensures"));
}

test "Phase1 Ora canonicalizer folds error constructor predicates" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @constructor_predicates(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) -> (i1, i1, !ora.int<256, false>, !ora.int<256, false>) {
        \\    %ok = ora.error.ok %arg0 : !ora.int<256, false> -> !ora.error_union<!ora.int<256, false>>
        \\    %err = ora.error.err %arg1 : !ora.int<256, false> -> !ora.error_union<!ora.int<256, false>>
        \\    %is_ok_err = ora.error.is_error %ok : !ora.error_union<!ora.int<256, false>> -> i1
        \\    %is_err_err = ora.error.is_error %err : !ora.error_union<!ora.int<256, false>> -> i1
        \\    %payload = ora.error.unwrap %ok : !ora.error_union<!ora.int<256, false>> -> !ora.int<256, false>
        \\    %error = ora.error.get_error %err : !ora.error_union<!ora.int<256, false>> -> !ora.int<256, false>
        \\    return %is_ok_err, %is_err_err, %payload, %error : i1, i1, !ora.int<256, false>, !ora.int<256, false>
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const predicates = try oraFunctionSlice(rendered, "constructor_predicates");
    try testing.expect(!std.mem.containsAtLeast(u8, predicates, 1, "ora.error.is_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, predicates, 1, "ora.error.unwrap"));
    try testing.expect(!std.mem.containsAtLeast(u8, predicates, 1, "ora.error.get_error"));
    try testing.expect(std.mem.containsAtLeast(u8, predicates, 1, "arith.constant false"));
    try testing.expect(std.mem.containsAtLeast(u8, predicates, 1, "arith.constant true"));
    try testing.expect(std.mem.containsAtLeast(u8, predicates, 1, "%arg0"));
    try testing.expect(std.mem.containsAtLeast(u8, predicates, 1, "%arg1"));
}

test "Phase1 Ora canonicalizer folds ADT constructor projections" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @constructor_projection(%arg0: !ora.int<256, false>) -> (!ora.int<256, false>, !ora.int<256, false>) {
        \\    %event = ora.adt.construct "Value"(%arg0) : (!ora.int<256, false>) -> !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>), ("Other", !ora.int<256, false>)>
        \\    %tag = ora.adt.tag %event : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>), ("Other", !ora.int<256, false>)> -> !ora.int<256, false>
        \\    %payload = ora.adt.payload %event, "Value" : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>), ("Other", !ora.int<256, false>)> -> !ora.int<256, false>
        \\    return %tag, %payload : !ora.int<256, false>, !ora.int<256, false>
        \\  }
        \\  func.func @kept_mismatched_payload(%arg0: !ora.int<256, false>) -> !ora.int<256, false> {
        \\    %event = ora.adt.construct "Value"(%arg0) : (!ora.int<256, false>) -> !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>), ("Other", !ora.int<256, false>)>
        \\    %payload = ora.adt.payload %event, "Other" : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>), ("Other", !ora.int<256, false>)> -> !ora.int<256, false>
        \\    return %payload : !ora.int<256, false>
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const projection = try oraFunctionSlice(rendered, "constructor_projection");
    try testing.expect(!std.mem.containsAtLeast(u8, projection, 1, "ora.adt.tag"));
    try testing.expect(!std.mem.containsAtLeast(u8, projection, 1, "ora.adt.payload"));
    try testing.expect(std.mem.containsAtLeast(u8, projection, 1, "arith.constant 1 : !ora.int<256, false>"));
    try testing.expect(std.mem.containsAtLeast(u8, projection, 1, "%arg0"));

    const mismatched = try oraFunctionSlice(rendered, "kept_mismatched_payload");
    try testing.expect(std.mem.containsAtLeast(u8, mismatched, 1, "ora.adt.payload"));
}

test "Phase1 Ora canonicalizer folds tuple extract from local tuple create" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @tuple_carrier(%arg0: !ora.int<256, false>, %arg1: i1) -> i1 {
        \\    %pair = ora.tuple_create(%arg0, %arg1) : !ora.int<256, false>, i1 -> !ora.tuple<!ora.int<256, false>, i1>
        \\    %picked = ora.tuple_extract %pair[1] : !ora.tuple<!ora.int<256, false>, i1> -> i1
        \\    return %picked : i1
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_create"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "return %arg1 : i1"));
}

test "Phase1 Ora canonicalizer folds tuple create from matching extracts" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @tuple_reconstruct(%arg0: !ora.tuple<!ora.int<256, false>, i1>) -> !ora.tuple<!ora.int<256, false>, i1> {
        \\    %left = ora.tuple_extract %arg0[0] : !ora.tuple<!ora.int<256, false>, i1> -> !ora.int<256, false>
        \\    %right = ora.tuple_extract %arg0[1] : !ora.tuple<!ora.int<256, false>, i1> -> i1
        \\    %rebuilt = ora.tuple_create(%left, %right) : !ora.int<256, false>, i1 -> !ora.tuple<!ora.int<256, false>, i1>
        \\    return %rebuilt : !ora.tuple<!ora.int<256, false>, i1>
        \\  }
        \\  func.func @tuple_permuted(%arg0: !ora.tuple<!ora.int<256, false>, i1>) -> !ora.tuple<i1, !ora.int<256, false>> {
        \\    %left = ora.tuple_extract %arg0[0] : !ora.tuple<!ora.int<256, false>, i1> -> !ora.int<256, false>
        \\    %right = ora.tuple_extract %arg0[1] : !ora.tuple<!ora.int<256, false>, i1> -> i1
        \\    %rebuilt = ora.tuple_create(%right, %left) : i1, !ora.int<256, false> -> !ora.tuple<i1, !ora.int<256, false>>
        \\    return %rebuilt : !ora.tuple<i1, !ora.int<256, false>>
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const reconstruct = try oraFunctionSlice(rendered, "tuple_reconstruct");
    try testing.expect(!std.mem.containsAtLeast(u8, reconstruct, 1, "ora.tuple_create"));
    try testing.expect(!std.mem.containsAtLeast(u8, reconstruct, 1, "ora.tuple_extract"));
    try testing.expect(std.mem.containsAtLeast(u8, reconstruct, 1, "return %arg0 : !ora.tuple<!ora.int<256, false>, i1>"));

    const permuted = try oraFunctionSlice(rendered, "tuple_permuted");
    try testing.expect(std.mem.containsAtLeast(u8, permuted, 1, "ora.tuple_create"));
}

test "Phase1 Ora canonicalizer folds struct field extract from local struct create" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  "ora.struct.decl"() ({
        \\  }) {name = "Pair", sym_name = "Pair", ora.field_names = ["left", "right"], ora.field_types = [!ora.int<256, false>, i1]} : () -> ()
        \\  func.func @struct_carrier(%arg0: !ora.int<256, false>, %arg1: i1) -> i1 {
        \\    %pair = ora.struct_init { %arg0, %arg1 } : !ora.int<256, false>, i1 -> !ora.struct<"Pair">
        \\    %picked = ora.struct_field_extract %pair, "right" : !ora.struct<"Pair"> -> i1
        \\    return %picked : i1
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.struct_field_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.struct_init"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "return %arg1 : i1"));
}

test "Phase1 Ora canonicalizer folds struct init from matching extracts" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  "ora.struct.decl"() ({
        \\  }) {name = "Pair", sym_name = "Pair", ora.field_names = ["left", "right"], ora.field_types = [!ora.int<256, false>, !ora.int<256, false>]} : () -> ()
        \\  func.func @struct_reconstruct(%arg0: !ora.struct<"Pair">) -> !ora.struct<"Pair"> {
        \\    %left = ora.struct_field_extract %arg0, "left" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %right = ora.struct_field_extract %arg0, "right" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %rebuilt = ora.struct_init { %left, %right } : !ora.int<256, false>, !ora.int<256, false> -> !ora.struct<"Pair">
        \\    return %rebuilt : !ora.struct<"Pair">
        \\  }
        \\  func.func @struct_instantiate_reconstruct(%arg0: !ora.struct<"Pair">) -> !ora.struct<"Pair"> {
        \\    %left = ora.struct_field_extract %arg0, "left" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %right = ora.struct_field_extract %arg0, "right" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %rebuilt = "ora.struct_instantiate"(%left, %right) {struct_name = "Pair"} : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.struct<"Pair">
        \\    return %rebuilt : !ora.struct<"Pair">
        \\  }
        \\  func.func @struct_instantiate_name_mismatch(%arg0: !ora.struct<"Pair">) -> !ora.struct<"Pair"> {
        \\    %left = ora.struct_field_extract %arg0, "left" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %right = ora.struct_field_extract %arg0, "right" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %rebuilt = "ora.struct_instantiate"(%left, %right) {struct_name = "Other"} : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.struct<"Pair">
        \\    return %rebuilt : !ora.struct<"Pair">
        \\  }
        \\  func.func @struct_swapped(%arg0: !ora.struct<"Pair">) -> !ora.struct<"Pair"> {
        \\    %left = ora.struct_field_extract %arg0, "left" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %right = ora.struct_field_extract %arg0, "right" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %rebuilt = ora.struct_init { %right, %left } : !ora.int<256, false>, !ora.int<256, false> -> !ora.struct<"Pair">
        \\    return %rebuilt : !ora.struct<"Pair">
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const reconstruct = try oraFunctionSlice(rendered, "struct_reconstruct");
    try testing.expect(!std.mem.containsAtLeast(u8, reconstruct, 1, "ora.struct_init"));
    try testing.expect(!std.mem.containsAtLeast(u8, reconstruct, 1, "ora.struct_field_extract"));
    try testing.expect(std.mem.containsAtLeast(u8, reconstruct, 1, "return %arg0 : !ora.struct<\"Pair\">"));

    const instantiate_reconstruct = try oraFunctionSlice(rendered, "struct_instantiate_reconstruct");
    try testing.expect(!std.mem.containsAtLeast(u8, instantiate_reconstruct, 1, "ora.struct_instantiate"));
    try testing.expect(!std.mem.containsAtLeast(u8, instantiate_reconstruct, 1, "ora.struct_field_extract"));
    try testing.expect(std.mem.containsAtLeast(u8, instantiate_reconstruct, 1, "return %arg0 : !ora.struct<\"Pair\">"));

    const instantiate_name_mismatch = try oraFunctionSlice(rendered, "struct_instantiate_name_mismatch");
    try testing.expect(std.mem.containsAtLeast(u8, instantiate_name_mismatch, 1, "ora.struct_instantiate"));

    const swapped = try oraFunctionSlice(rendered, "struct_swapped");
    try testing.expect(std.mem.containsAtLeast(u8, swapped, 1, "ora.struct_init"));
}

test "Phase1 Ora canonicalizer folds struct field extract through field update" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  "ora.struct.decl"() ({
        \\  }) {name = "Pair", sym_name = "Pair", ora.field_names = ["left", "right"], ora.field_types = [!ora.int<256, false>, i1]} : () -> ()
        \\  func.func @updated_field(%arg0: !ora.int<256, false>, %arg1: i1, %arg2: !ora.int<256, false>) -> (!ora.int<256, false>, i1) {
        \\    %pair = ora.struct_init { %arg0, %arg1 } : !ora.int<256, false>, i1 -> !ora.struct<"Pair">
        \\    %updated = ora.struct_field_update %pair, "left", %arg2 : !ora.struct<"Pair">, !ora.int<256, false> -> !ora.struct<"Pair">
        \\    %left = ora.struct_field_extract %updated, "left" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %right = ora.struct_field_extract %updated, "right" : !ora.struct<"Pair"> -> i1
        \\    return %left, %right : !ora.int<256, false>, i1
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.struct_field_update"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.struct_field_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.struct_init"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "return %arg2, %arg1 : !ora.int<256, false>, i1"));
}

test "Phase1 Ora canonicalizer folds struct field update into constructors" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  "ora.struct.decl"() ({
        \\  }) {name = "Pair", sym_name = "Pair", ora.field_names = ["left", "right"], ora.field_types = [!ora.int<256, false>, i1]} : () -> ()
        \\  func.func @init_update(%arg0: !ora.int<256, false>, %arg1: i1, %arg2: !ora.int<256, false>) -> !ora.struct<"Pair"> {
        \\    %pair = ora.struct_init { %arg0, %arg1 } : !ora.int<256, false>, i1 -> !ora.struct<"Pair">
        \\    %updated = ora.struct_field_update %pair, "left", %arg2 : !ora.struct<"Pair">, !ora.int<256, false> -> !ora.struct<"Pair">
        \\    return %updated : !ora.struct<"Pair">
        \\  }
        \\  func.func @instantiate_update(%arg0: !ora.int<256, false>, %arg1: i1, %arg2: i1) -> !ora.struct<"Pair"> {
        \\    %pair = "ora.struct_instantiate"(%arg0, %arg1) {struct_name = "Pair"} : (!ora.int<256, false>, i1) -> !ora.struct<"Pair">
        \\    %updated = ora.struct_field_update %pair, "right", %arg2 : !ora.struct<"Pair">, i1 -> !ora.struct<"Pair">
        \\    return %updated : !ora.struct<"Pair">
        \\  }
        \\  func.func @instantiate_name_mismatch(%arg0: !ora.int<256, false>, %arg1: i1, %arg2: i1) -> !ora.struct<"Pair"> {
        \\    %pair = "ora.struct_instantiate"(%arg0, %arg1) {struct_name = "Other"} : (!ora.int<256, false>, i1) -> !ora.struct<"Pair">
        \\    %updated = ora.struct_field_update %pair, "right", %arg2 : !ora.struct<"Pair">, i1 -> !ora.struct<"Pair">
        \\    return %updated : !ora.struct<"Pair">
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const init_update = try oraFunctionSlice(rendered, "init_update");
    try testing.expect(!std.mem.containsAtLeast(u8, init_update, 1, "ora.struct_field_update"));
    try testing.expect(std.mem.containsAtLeast(u8, init_update, 1, "ora.struct_init{"));
    try testing.expect(std.mem.containsAtLeast(u8, init_update, 1, "%arg2, %arg1"));

    const instantiate_update = try oraFunctionSlice(rendered, "instantiate_update");
    try testing.expect(!std.mem.containsAtLeast(u8, instantiate_update, 1, "ora.struct_field_update"));
    try testing.expect(!std.mem.containsAtLeast(u8, instantiate_update, 1, "ora.struct_instantiate"));
    try testing.expect(std.mem.containsAtLeast(u8, instantiate_update, 1, "ora.struct_init{"));
    try testing.expect(std.mem.containsAtLeast(u8, instantiate_update, 1, "%arg0, %arg2"));

    const instantiate_name_mismatch = try oraFunctionSlice(rendered, "instantiate_name_mismatch");
    try testing.expect(std.mem.containsAtLeast(u8, instantiate_name_mismatch, 1, "ora.struct_instantiate"));
    try testing.expect(std.mem.containsAtLeast(u8, instantiate_name_mismatch, 1, "ora.struct_field_update"));
}

test "Phase1 Ora canonicalizer folds redundant struct field updates" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  "ora.struct.decl"() ({
        \\  }) {name = "Pair", sym_name = "Pair", ora.field_names = ["left", "right"], ora.field_types = [!ora.int<256, false>, i1]} : () -> ()
        \\  func.func @overwrite(%arg0: !ora.struct<"Pair">, %arg1: !ora.int<256, false>, %arg2: !ora.int<256, false>) -> !ora.struct<"Pair"> {
        \\    %first = ora.struct_field_update %arg0, "left", %arg1 : !ora.struct<"Pair">, !ora.int<256, false> -> !ora.struct<"Pair">
        \\    %second = ora.struct_field_update %first, "left", %arg2 : !ora.struct<"Pair">, !ora.int<256, false> -> !ora.struct<"Pair">
        \\    return %second : !ora.struct<"Pair">
        \\  }
        \\  func.func @noop(%arg0: !ora.struct<"Pair">) -> !ora.struct<"Pair"> {
        \\    %left = ora.struct_field_extract %arg0, "left" : !ora.struct<"Pair"> -> !ora.int<256, false>
        \\    %same = ora.struct_field_update %arg0, "left", %left : !ora.struct<"Pair">, !ora.int<256, false> -> !ora.struct<"Pair">
        \\    return %same : !ora.struct<"Pair">
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const overwrite = try oraFunctionSlice(rendered, "overwrite");
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, overwrite, "ora.struct_field_update"));
    try testing.expect(std.mem.containsAtLeast(u8, overwrite, 1, "ora.struct_field_update %arg0, \"left\", %arg2"));

    const noop = try oraFunctionSlice(rendered, "noop");
    try testing.expect(!std.mem.containsAtLeast(u8, noop, 1, "ora.struct_field_update"));
    try testing.expect(!std.mem.containsAtLeast(u8, noop, 1, "ora.struct_field_extract"));
    try testing.expect(std.mem.containsAtLeast(u8, noop, 1, "return %arg0 : !ora.struct<\"Pair\">"));
}

test "Phase1 Ora CSE removes duplicate pure ops but keeps gas reads" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @pure_duplicate(%arg0: i256, %arg1: i256) -> (i256, i256) {
        \\    %first = arith.addi %arg0, %arg1 : i256
        \\    %second = arith.addi %arg0, %arg1 : i256
        \\    return %first, %second : i256, i256
        \\  }
        \\  func.func @gas_reads() -> (i256, i256) {
        \\    %first = ora.evm.gas : i256
        \\    %second = ora.evm.gas : i256
        \\    return %first, %second : i256, i256
        \\  }
        \\  func.func @external_calls(%target: !ora.address, %gas: !ora.int<256, false>, %calldata: !ora.bytes) -> (i1, !ora.bytes, i1, !ora.bytes) {
        \\    %success0, %returndata0 = ora.external_call %target, %gas, %calldata {call_kind = "call", method_name = "ping", trait_name = "Remote"} : !ora.address, !ora.int<256, false>, !ora.bytes -> i1, !ora.bytes
        \\    %success1, %returndata1 = ora.external_call %target, %gas, %calldata {call_kind = "call", method_name = "ping", trait_name = "Remote"} : !ora.address, !ora.int<256, false>, !ora.bytes -> i1, !ora.bytes
        \\    return %success0, %returndata0, %success1, %returndata1 : i1, !ora.bytes, i1, !ora.bytes
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "arith.addi"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "ora.evm.gas"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "ora.external_call"));
}

test "Phase1 Ora canonicalization removes dead pure values only" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @dead_pure(%arg0: i256) -> i256 {
        \\    %dead = arith.addi %arg0, %arg0 : i256
        \\    return %arg0 : i256
        \\  }
        \\  func.func @keeps_effectful(%arg0: !ora.int<256, false>) -> !ora.int<256, false> {
        \\    %gas = ora.evm.gas : !ora.int<256, false>
        \\    return %arg0 : !ora.int<256, false>
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const dead_pure = try oraFunctionSlice(rendered, "dead_pure");
    try testing.expect(!std.mem.containsAtLeast(u8, dead_pure, 1, "arith.addi"));

    const keeps_effectful = try oraFunctionSlice(rendered, "keeps_effectful");
    try testing.expect(std.mem.containsAtLeast(u8, keeps_effectful, 1, "ora.evm.gas"));
}

test "Phase1 Ora canonicalization preserves public ABI arguments" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @public_unused_arg(%used: !ora.int<256, false>, %unused: !ora.int<256, false>) -> !ora.int<256, false> attributes {ora.visibility = "pub"} {
        \\    return %used : !ora.int<256, false>
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const public_fn = try oraFunctionSlice(rendered, "public_unused_arg");
    const header_end = std.mem.indexOf(u8, public_fn, "{") orelse return error.TestUnexpectedResult;
    const public_fn_header = public_fn[0..header_end];
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, public_fn_header, "!ora.int<256, false>"));
}

test "Phase1 storage read CSE respects storage write barriers" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.global "counter" : !ora.int<256, false>
        \\  ora.global "other" : !ora.int<256, false>
        \\  ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\  func.func @same_block() -> (!ora.int<256, false>, !ora.int<256, false>) {
        \\    %first = ora.sload "counter" : !ora.int<256, false>
        \\    %second = ora.sload "counter" : !ora.int<256, false>
        \\    return %first, %second : !ora.int<256, false>, !ora.int<256, false>
        \\  }
        \\  func.func @same_global_store_barrier(%value: !ora.int<256, false>) -> (!ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>) {
        \\    %counter_before = ora.sload "counter" : !ora.int<256, false>
        \\    %other_before = ora.sload "other" : !ora.int<256, false>
        \\    ora.sstore %value, "counter" : !ora.int<256, false>
        \\    %counter_after = ora.sload "counter" : !ora.int<256, false>
        \\    %other_after = ora.sload "other" : !ora.int<256, false>
        \\    return %counter_before, %counter_after, %other_before, %other_after : !ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>
        \\  }
        \\  func.func @map_store_barrier(%key: !ora.int<256, false>, %value: !ora.int<256, false>) -> (!ora.int<256, false>, !ora.int<256, false>) {
        \\    %before = ora.sload "counter" : !ora.int<256, false>
        \\    %map = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    ora.map_store %map, %key, %value : !ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>, !ora.int<256, false>
        \\    %after = ora.sload "counter" : !ora.int<256, false>
        \\    return %before, %after : !ora.int<256, false>, !ora.int<256, false>
        \\  }
        \\  func.func @external_call_barrier(%target: !ora.address, %gas: !ora.int<256, false>, %calldata: !ora.bytes) -> (!ora.int<256, false>, !ora.int<256, false>, i1, !ora.bytes) {
        \\    %before = ora.sload "counter" : !ora.int<256, false>
        \\    %success, %returndata = ora.external_call %target, %gas, %calldata {call_kind = "call", method_name = "ping", trait_name = "Remote"} : !ora.address, !ora.int<256, false>, !ora.bytes -> i1, !ora.bytes
        \\    %after = ora.sload "counter" : !ora.int<256, false>
        \\    return %before, %after, %success, %returndata : !ora.int<256, false>, !ora.int<256, false>, i1, !ora.bytes
        \\  }
        \\  func.func @gas_read_barrier() -> (!ora.int<256, false>, !ora.int<256, false>, i256) {
        \\    %before = ora.sload "counter" : !ora.int<256, false>
        \\    %gas = ora.evm.gas : i256
        \\    %after = ora.sload "counter" : !ora.int<256, false>
        \\    return %before, %after, %gas : !ora.int<256, false>, !ora.int<256, false>, i256
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraCanonicalizeOraMLIR(ctx, module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    const same_block = try oraFunctionSlice(rendered, "same_block");
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, same_block, "ora.sload \"counter\""));

    const store_barrier = try oraFunctionSlice(rendered, "same_global_store_barrier");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, store_barrier, "ora.sload \"counter\""));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, store_barrier, "ora.sload \"other\""));

    const map_barrier = try oraFunctionSlice(rendered, "map_store_barrier");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, map_barrier, "ora.sload \"counter\""));

    const external_call_barrier = try oraFunctionSlice(rendered, "external_call_barrier");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, external_call_barrier, "ora.sload \"counter\""));

    const gas_read_barrier = try oraFunctionSlice(rendered, "gas_read_barrier");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, gas_read_barrier, "ora.sload \"counter\""));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gas_read_barrier, "ora.evm.gas"));
}

test "OraToSIR keeps arithmetic width-changing carrier round trips" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @add_carrier(%arg0: !sir.u256, %arg1: !sir.u256) {
        \\    %0 = sir.add %arg0 : !sir.u256, %arg1 : !sir.u256 : !sir.u256
        \\    %1 = sir.bitcast %0 : !sir.u256 : i128
        \\    %2 = sir.bitcast %1 : i128 : !sir.u256
        \\    sir.iret %2
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

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "sir.bitcast"));
}

test "OraToSIR handles residual cast worklist entries erased by an earlier cast" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @stale_residual_cast(%arg0: !sir.u256) {
        \\    %0 = builtin.unrealized_conversion_cast %arg0 : !sir.u256 to i128
        \\    %1 = builtin.unrealized_conversion_cast %0 : i128 to !sir.u256
        \\    %2 = builtin.unrealized_conversion_cast %0 : i128 to !sir.u256
        \\    %3 = sir.add %1 : !sir.u256, %2 : !sir.u256 : !sir.u256
        \\    sir.iret %3
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

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "builtin.unrealized_conversion_cast"));
}

test "OraToSIR masks unsigned materialization widths above u64" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @mask_u128(%arg0: i128) {
        \\    %0 = builtin.unrealized_conversion_cast %arg0 : i128 to !sir.u256
        \\    sir.iret %0
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

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "builtin.unrealized_conversion_cast"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "340282366920938463463374607431768211455"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.and"));
}

test "OraToSIR rejects generic aggregate to scalar cast fallback" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @bad(%arg0: !ora.string) {
        \\    %0 = builtin.unrealized_conversion_cast %arg0 : !ora.string to !sir.u256
        \\    sir.iret %0
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(!mlir.oraConvertToSIR(ctx, module, false));
}

test "OraToSIR rejects value typed try_stmt with empty yield" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  func.func @bad() -> i256 {
        \\    %0 = "ora.try_stmt"() ({
        \\      ora.yield
        \\    }, {
        \\      ora.yield
        \\    }) : () -> i256
        \\    ora.return %0 : i256
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(!mlir.oraConvertToSIR(ctx, module, false));
}

test "OraToSIR keeps narrow signed arithmetic in u256 carrier" {
    const source_text =
        \\pub fn div_i128(a: i128, b: i128) -> i128 {
        \\    return a / b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.sdiv"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.bitcast"));
}

test "OraToSIR preserves explicit narrow integer truncation mask" {
    const source_text =
        \\pub fn narrow(big: i256) -> i8 {
        \\    return @cast(i8, big);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const narrow_fn = try functionSlice(rendered, "narrow");

    try testing.expect(std.mem.containsAtLeast(u8, narrow_fn, 1, "const 0xFF"));
    try testing.expect(std.mem.containsAtLeast(u8, narrow_fn, 1, "and"));
    try testing.expect(!std.mem.containsAtLeast(u8, narrow_fn, 1, "bitcast"));
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

test "compiler converts runtime abiDecode scalar memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_scalar(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(u256, payload);
        \\        return match (decoded) {
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
    const decode_fn = try functionSlice(rendered, "decode_scalar");

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_scalar:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bool memory result with canonical validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bool(payload: bytes) -> bool {
        \\        let decoded = @abiDecode(bool, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => false,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_bool:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "or"));
    const decode_fn = try functionSlice(rendered, "decode_bool");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecodePermissive memory result shapes through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_scalar(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive(u8, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_text(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive(string, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_values(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive(slice[u256], payload);
        \\        return match (decoded) {
        \\            Ok(values) => values[0],
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_text(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive((u256, string), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    try expectNoResidualOraRuntimeOps(rendered);

    const scalar_fn = try functionSlice(rendered, "decode_scalar");
    const text_fn = try functionSlice(rendered, "decode_text");
    const values_fn = try functionSlice(rendered, "decode_values");
    const pair_text_fn = try functionSlice(rendered, "decode_pair_text");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(scalar_fn);
    try expectDynamicAbiDecodeWordGuardChain(text_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, text_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(values_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_text_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, pair_text_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_text_fn);
}

test "compiler converts runtime abiDecodePermissive mixed dynamic hex literal through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode() -> u256 {
        \\        let decoded = @abiDecodePermissive((u256, string), hex"000000000000000000000000000000000000000000000000000000000000000700000000000000000000000000000000000000000000000000000000000000600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000161ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        \\        return match (decoded) {
        \\            Ok(value) => value.0 + @cast(u256, value.1[0]),
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
}

test "compiler abiDecode N3b4 validates public calldata bool and address params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn accept(flag: bool, owner: address, amount: u8, delta: i8, tag: bytes4) -> bool {
        \\        return true;
        \\    }
        \\    pub fn full(signed: i256, tag: bytes32) -> bool {
        \\        return true;
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
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "or"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 5, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "shl"));
    try expectOrderedNeedles(main_fn, &.{ "a_accept = calldataload", "eq a_accept", "eq a_accept", "or" });
    try expectOrderedNeedles(main_fn, &.{ "b_accept = calldataload", "and b_accept", "eq b_accept" });
    try expectOrderedNeedles(main_fn, &.{ "n_accept = calldataload", "and n_accept", "eq n_accept" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept = calldataload", "signextend", "eq arg_accept" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept_0 = calldataload", "shr", "shl", "eq arg_accept_0" });
    try expectOrderedNeedles(main_fn, &.{ "a_accept = calldataload", "or", ": @abi_decode_revert_4", "b_accept = calldataload" });
    try expectOrderedNeedles(main_fn, &.{ "b_accept = calldataload", "eq b_accept", ": @abi_decode_revert_5", "n_accept = calldataload" });
    try expectOrderedNeedles(main_fn, &.{ "n_accept = calldataload", "eq n_accept", ": @abi_decode_revert_3" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept = calldataload", "eq arg_accept", ": @abi_decode_revert_3" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept_0 = calldataload", "eq arg_accept_0", "? @accept_exec : @abi_decode_revert_6" });
    const accept_full_call = std.mem.indexOf(u8, main_fn, "icall @full") orelse return error.TestUnexpectedResult;
    const accept_full_label = std.mem.lastIndexOf(u8, main_fn[0..accept_full_call], "full_exec") orelse return error.TestUnexpectedResult;
    const accept_full_slice = main_fn[accept_full_label..accept_full_call];
    try testing.expect(std.mem.containsAtLeast(u8, accept_full_slice, 2, "calldataload"));
    try testing.expect(!std.mem.containsAtLeast(u8, accept_full_slice, 1, "signextend"));
    try testing.expect(!std.mem.containsAtLeast(u8, accept_full_slice, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, accept_full_slice, 1, "shl"));
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_3", "const 0x3", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_4", "const 0x4", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_5", "const 0x5", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_6", "const 0x6", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "accept_exec"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept a_accept b_accept"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept a_accept b_accept n_accept"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept a_accept b_accept n_accept arg_accept"));
}

test "compiler decodePermissive marker relaxes public calldata canonicality checks" {
    const source_text =
        \\contract Entry {
        \\    @decodePermissive
        \\    pub fn accept(flag: bool, owner: address, note: string) -> u256 {
        \\        if (flag) {
        \\            return note[0];
        \\        }
        \\        return @abiEncode(owner)[31];
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
    const main_fn = try functionSlice(rendered, "main");

    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_4"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_5"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_11"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "mload8"));
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_14", "const 0xE", "mstore256", "revert" });
}

test "compiler abiDecode N3b4 validates public calldata enum range before call" {
    const source_text =
        \\enum Status: u8 { Active, Paused }
        \\contract Entry {
        \\    pub fn set(status: Status, flag: bool) -> bool {
        \\        return true;
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
    try expectOrderedNeedles(main_fn, &.{ "a_set = calldataload", "and a_set", "eq a_set", "lt" });
    try expectOrderedNeedles(main_fn, &.{ "eq a_set", ": @abi_decode_revert_3" });
    try expectOrderedNeedles(main_fn, &.{ "lt", ": @abi_decode_revert_7" });
    try expectOrderedNeedles(main_fn, &.{ "b_set = calldataload", "eq b_set", "eq b_set", "or" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_3", "const 0x3", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_4", "const 0x4", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_7", "const 0x7", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "set_exec"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @set"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @set a_set b_set"));
}

test "compiler abiDecode N3b4 validates public calldata refinements before call" {
    const source_text =
        \\type PositiveByte = MinValue<u8, 1>;
        \\type SignedFloor = MinValue<i8, -5>;
        \\contract Entry {
        \\    pub fn check(amount: PositiveByte, delta: SignedFloor, owner: NonZeroAddress) -> bool {
        \\        return true;
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
    try expectOrderedNeedles(main_fn, &.{ "a_check = calldataload", "and a_check", "eq a_check", "lt", "iszero", ": @abi_decode_revert_3", ": @abi_decode_revert_10" });
    try expectOrderedNeedles(main_fn, &.{ "b_check = calldataload", "signextend", "eq b_check", "slt", "iszero", ": @abi_decode_revert_3", ": @abi_decode_revert_10" });
    try expectOrderedNeedles(main_fn, &.{ "n_check = calldataload", "and n_check", "eq n_check", "eq", "iszero", ": @abi_decode_revert_5", ": @abi_decode_revert_10" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_3", "const 0x3", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_5", "const 0x5", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_10", "const 0xA", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "check_exec"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @check"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @check a_check b_check n_check"));
}

test "compiler abiDecode N3b4 validates public calldata string and bytes params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_text(text: string) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_data(data: bytes) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_both(text: string, data: bytes) -> bool {
        \\        return true;
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
    for ([_][]const u8{ "take_text", "take_data" }) |name| {
        var load_buf: [64]u8 = undefined;
        var exec_buf: [64]u8 = undefined;
        const load = try std.fmt.bufPrint(&load_buf, "a_{s} = calldataload", .{name});
        const exec = try std.fmt.bufPrint(&exec_buf, "? @{s}_exec : @abi_decode_revert_11", .{name});
        try expectOrderedNeedles(main_fn, &.{
            load,
            "eq a_",
            ": @abi_decode_revert_11",
            "calldatasize",
            ": @abi_decode_revert_0",
            "calldataload",
            "large_const 0xFFFFFFFFFFFFFFFF",
            "gt",
            ": @abi_decode_revert_13",
            "const 0x100000",
            "gt",
            ": @abi_decode_revert_14",
            "div",
            "mul",
            ": @abi_decode_revert_0",
            "calldatacopy",
            "mload8",
            "eq",
            exec,
        });
    }
    try expectOrderedNeedles(main_fn, &.{
        "a_take_both = calldataload",
        "eq a_take_both",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_both = calldataload",
        "eq b_take_both",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x100000",
        "gt",
        ": @abi_decode_revert_14",
        "div",
        "mul",
        ": @abi_decode_revert_0",
        "calldatacopy",
        "mload8",
        "eq",
        "? @take_both_exec : @abi_decode_revert_11",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_14", "const 0xE", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_text"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_data"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_both"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_text a_take_text"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_data a_take_data"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_both_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata u256 slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_values(values: slice[u256]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_values(text: string, values: slice[u256]) -> bool {
        \\        return true;
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
    try expectOrderedNeedles(main_fn, &.{
        "a_take_values = calldataload",
        "eq a_take_values",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @take_values_exec : @abi_decode_revert_0",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_values = calldataload",
        "eq a_take_text_values",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_values = calldataload",
        "eq b_take_text_values",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @take_text_values_exec : @abi_decode_revert_0",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_values_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_values_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata address slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_addresses(values: slice[address]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_addresses(text: string, values: slice[address]) -> bool {
        \\        return true;
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
    try expectOrderedNeedles(main_fn, &.{
        "a_take_addresses = calldataload",
        "eq a_take_addresses",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "and",
        "eq",
        "? @take_addresses_exec : @abi_decode_revert_5",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_addresses = calldataload",
        "eq a_take_text_addresses",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_addresses = calldataload",
        "eq b_take_text_addresses",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "and",
        "eq",
        "? @take_text_addresses_exec : @abi_decode_revert_5",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_5", "const 0x5", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_addresses_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_addresses_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata bool slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_bools(values: slice[bool]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_bools(text: string, values: slice[bool]) -> bool {
        \\        return true;
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
    try expectOrderedNeedles(main_fn, &.{
        "a_take_bools = calldataload",
        "eq a_take_bools",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "eq",
        "eq",
        "or",
        "? @take_bools_exec : @abi_decode_revert_4",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_bools = calldataload",
        "eq a_take_text_bools",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_bools = calldataload",
        "eq b_take_text_bools",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "eq",
        "eq",
        "or",
        "? @take_text_bools_exec : @abi_decode_revert_4",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_4", "const 0x4", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_bools_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_bools_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata fixed bytes slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_tags(values: slice[bytes4]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_tags(text: string, values: slice[bytes4]) -> bool {
        \\        return true;
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
    try expectOrderedNeedles(main_fn, &.{
        "a_take_tags = calldataload",
        "eq a_take_tags",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "shr",
        "shl",
        "eq",
        ": @abi_decode_revert_6",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_tags = calldataload",
        "eq a_take_text_tags",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_tags = calldataload",
        "eq b_take_text_tags",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "shr",
        "shl",
        "eq",
        ": @abi_decode_revert_6",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_6", "const 0x6", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_tags"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_text_tags"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_tags_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_tags_exec : @revert_error"));
}

test "compiler abiDecode N3b4 rejects unsupported nested dynamic calldata arrays before legacy fallback" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_nested(values: slice[slice[u256]]) -> bool {
        \\        return true;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(!mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));
}

test "compiler abiDecode N3b4 validates public calldata dynamic tuple params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn value(t: (u256, string)) -> u256 {
        \\        return t.0 + t.1.len;
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
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "value_"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldataload"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_11"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "calldatacopy"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "mload8"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @value"));
}

test "compiler abiDecode N3b5 validates dynamic constructor string and bytes calldata" {
    const cases = [_][]const u8{
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(name: string) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(payload: bytes) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(name: string, payload: bytes) {
        \\        touched = 9;
        \\    }
        \\}
        ,
    };

    for (cases) |source_text| {
        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
        try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

        const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
        defer testing.allocator.free(rendered);

        const init_fn = try functionSlice(rendered, "init");
        try expectOrderedNeedles(init_fn, &.{ "codesize", "init_end", "codecopy", "mload256", "eq", ": @init_abi_decode_revert_11" });
        try expectOrderedNeedles(init_fn, &.{ "mload256", "gt", ": @init_abi_decode_revert_13" });
        try expectOrderedNeedles(init_fn, &.{ "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @init_abi_decode_revert_13", "const 0x100000", "gt", ": @init_abi_decode_revert_14", "div", "mul", ": @init_abi_decode_revert_0", "mload8" });
        try expectOrderedNeedles(init_fn, &.{ "mload8", "eq", "? @", ": @init_abi_decode_revert_11" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_14", "const 0xE", "mstore256", "revert" });
        try testing.expect(std.mem.containsAtLeast(u8, init_fn, 1, "icall @__ora_user_init"));
    }
}

test "compiler abiDecode N3b5 validates dynamic constructor slice calldata" {
    const cases = [_][]const u8{
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(values: slice[u256]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(owners: slice[address]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(flags: slice[bool]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(tags: slice[bytes4]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(name: string, owners: slice[address]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
    };

    for (cases) |source_text| {
        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
        try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

        const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
        defer testing.allocator.free(rendered);

        const init_fn = try functionSlice(rendered, "init");
        try expectOrderedNeedles(init_fn, &.{ "codesize", "init_end", "codecopy", "mload256", "eq", ": @init_abi_decode_revert_11" });
        try expectOrderedNeedles(init_fn, &.{ "mload256", "gt", ": @init_abi_decode_revert_13" });
        try expectOrderedNeedles(init_fn, &.{ "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @init_abi_decode_revert_13", "const 0x8000", "gt", ": @init_abi_decode_revert_9", "mul", ": @init_abi_decode_revert_0" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
        try testing.expect(std.mem.containsAtLeast(u8, init_fn, 1, "icall @__ora_user_init"));
    }
}

test "compiler abiDecode N3b5 validates dynamic Result carrier bounds before payload loads" {
    const cases = [_]struct {
        source: []const u8,
        cap: []const u8,
        cap_error: []const u8,
        payload_needles: []const []const u8,
    }{
        .{
            .source =
            \\error Failure(code: u256);
            \\
            \\contract Entry {
            \\    pub fn consume(value: Result<bytes, Failure>) -> u256 {
            \\        return match (value) {
            \\            Ok(inner) => inner.len,
            \\            Err(err) => err.code,
            \\        };
            \\    }
            \\}
            ,
            .cap = "const 0x100000",
            .cap_error = ": @abi_decode_revert_14",
            .payload_needles = &.{ ": @abi_decode_revert_0", "calldataload", "shr", "? @consume_exec : @abi_decode_revert_11" },
        },
        .{
            .source =
            \\error Failure(code: u256);
            \\
            \\contract Entry {
            \\    pub fn consume(value: Result<slice[address], Failure>) -> u256 {
            \\        return match (value) {
            \\            Ok(inner) => 1,
            \\            Err(err) => err.code,
            \\        };
            \\    }
            \\}
            ,
            .cap = "const 0x8000",
            .cap_error = ": @abi_decode_revert_9",
            .payload_needles = &.{ ": @abi_decode_revert_0", "calldataload", "and", "eq", "? @consume_exec : @abi_decode_revert_5" },
        },
    };

    for (cases) |case| {
        var compilation = try compileText(case.source);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
        try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

        const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
        defer testing.allocator.free(rendered);

        const main_fn = try functionSlice(rendered, "main");
        try expectOrderedNeedles(main_fn, &.{ "calldatasize", ": @abi_decode_revert_0", "calldataload", "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @abi_decode_revert_13" });
        try expectOrderedNeedles(main_fn, &.{ "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @abi_decode_revert_13", case.cap, "gt", case.cap_error, "mul", ": @abi_decode_revert_0" });
        try expectOrderedNeedles(main_fn, case.payload_needles);
        try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
        try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
        try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
        try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @consume"));
        try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @consume_exec : @revert_error"));
    }
}

test "compiler converts runtime abiDecode bool oversize priority through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bool_priority() -> u256 {
        \\        let decoded = @abiDecode(bool, hex"00000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000");
        \\        return match (decoded) {
        \\            Ok(_) => 99,
        \\            Err(_) => 1,
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
    const decode_fn = try functionSlice(rendered, "decode_bool_priority");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    // The oversize branch must be gated by the decoded Result tag so an earlier
    // decode error (invalid bool here) is not overwritten by oversize_buffer.
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "iszero"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "0x4"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "0x1"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode u8 memory result with canonical padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_u8(payload: bytes) -> u8 {
        \\        let decoded = @abiDecode(u8, payload);
        \\        return match (decoded) {
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_u8:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_u8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode address memory result with canonical prefix validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_address(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(address, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_address:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_address");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bounds refinements with refinement validation" {
    const source_text =
        \\type PositiveAmount = MinValue<u256, 1>;
        \\type SmallAmount = MaxValue<u256, 10>;
        \\type RangedAmount = InRange<u256, 2, 8>;
        \\type PositiveByte = MinValue<u8, 1>;
        \\type SignedFloor = MinValue<i8, -5>;
        \\type SignedNonNegative = MinValue<i8, 0>;
        \\type NestedAmount = MinValue<MaxValue<u256, 10>, 1>;
        \\contract Decode {
        \\    pub fn decode_positive(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(PositiveAmount, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_small(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(SmallAmount, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_range(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(RangedAmount, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_positive_u8(payload: bytes) -> u8 {
        \\        let decoded = @abiDecode(PositiveByte, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_signed_i8(payload: bytes) -> i8 {
        \\        let decoded = @abiDecode(SignedFloor, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_signed_i8_zero(payload: bytes) -> i8 {
        \\        let decoded = @abiDecode(SignedNonNegative, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_nested(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(NestedAmount, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
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
    const decode_fn = try functionSlice(rendered, "decode_positive");

    // Full-word u256 canonical decode emits no lt/gt comparison. Refined
    // targets now also get static length checks, so counts below distinguish
    // refinement comparisons from the length-check comparisons.
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));

    const max_decode_fn = try functionSlice(rendered, "decode_small");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(max_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, max_decode_fn, 1, "bitcast"));

    const range_decode_fn = try functionSlice(rendered, "decode_range");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(range_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 1, "and"));
    try testing.expect(!std.mem.containsAtLeast(u8, range_decode_fn, 1, "bitcast"));

    const positive_u8_decode_fn = try functionSlice(rendered, "decode_positive_u8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(positive_u8_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "bitcast"));

    const signed_i8_decode_fn = try functionSlice(rendered, "decode_signed_i8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(signed_i8_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "slt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "bitcast"));

    const signed_i8_zero_decode_fn = try functionSlice(rendered, "decode_signed_i8_zero");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(signed_i8_zero_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "slt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "bitcast"));

    const nested_decode_fn = try functionSlice(rendered, "decode_nested");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(nested_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 1, "and"));
    try testing.expect(!std.mem.containsAtLeast(u8, nested_decode_fn, 1, "bitcast"));
}

test "compiler converts runtime abiDecode NonZeroAddress memory result with refinement validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_owner(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(NonZeroAddress, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
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
    const decode_fn = try functionSlice(rendered, "decode_owner");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode i8 memory result with canonical sign extension" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_i8(payload: bytes) -> i8 {
        \\        let decoded = @abiDecode(i8, payload);
        \\        return match (decoded) {
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_i8:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_i8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode i256 memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_i256(payload: bytes) -> i256 {
        \\        let decoded = @abiDecode(i256, payload);
        \\        return match (decoded) {
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
    const decode_fn = try functionSlice(rendered, "decode_i256");

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_i256:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "signextend"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode fixed bytes memory result with canonical padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bytes4(payload: bytes) -> bytes4 {
        \\        let decoded = @abiDecode(bytes4, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => hex"00000000",
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_bytes4:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_bytes4");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bytes1 memory result with canonical padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bytes1(payload: bytes) -> bytes1 {
        \\        let decoded = @abiDecode(bytes1, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => hex"00",
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
    const decode_fn = try functionSlice(rendered, "decode_bytes1");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bytes32 memory result without padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bytes32(payload: bytes) -> bytes32 {
        \\        let decoded = @abiDecode(bytes32, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => hex"0000000000000000000000000000000000000000000000000000000000000000",
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
    const decode_fn = try functionSlice(rendered, "decode_bytes32");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode enum memory result with range validation" {
    const source_text =
        \\enum Status: u8 { Active, Paused }
        \\contract Decode {
        \\    pub fn decode_status(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(Status, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
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
    const decode_fn = try functionSlice(rendered, "decode_status");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bitfield memory result with declared width validation" {
    const source_text =
        \\bitfield Flags: u8 {
        \\    enabled: bool @0;
        \\    mode: u7 @1;
        \\}
        \\contract Decode {
        \\    pub fn decode_flags(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(Flags, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
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
    const decode_fn = try functionSlice(rendered, "decode_flags");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode void memory result with empty bytes validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_void(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(void, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
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
    const decode_fn = try functionSlice(rendered, "decode_void");

    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode dynamic string and bytes memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_string(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(string, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_bytes(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(bytes, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_values(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[u256], payload);
        \\        return match (decoded) {
        \\            Ok(values) => values[0] + values[1],
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_addresses(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[address], payload);
        \\        match (decoded) {
        \\            Ok(values) => {
        \\                if (values[0] == 0x0000000000000000000000000000000000000001) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_bools(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bool], payload);
        \\        match (decoded) {
        \\            Ok(values) => {
        \\                if (values[0]) {
        \\                    if (values[1]) {
        \\                        return 2;
        \\                    }
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_tags(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bytes4], payload);
        \\        match (decoded) {
        \\            Ok(tags) => {
        \\                const expected: bytes4 = hex"aabbccdd";
        \\                if (tags[0] == expected) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_tag_one(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bytes1], payload);
        \\        match (decoded) {
        \\            Ok(tags) => {
        \\                const expected: bytes1 = hex"aa";
        \\                if (tags[0] == expected) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_tag_full(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bytes32], payload);
        \\        match (decoded) {
        \\            Ok(tags) => {
        \\                const expected: bytes32 = hex"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        \\                if (tags[0] == expected) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_text(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, string), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_bytes(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, bytes), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0 + @cast(u256, value.1[0]),
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_values(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[u256]), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0 + @cast(u256, value.1[0]) + value.1[1],
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_addresses(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[address]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                if (value.1[0] == 0x0000000000000000000000000000000000000001) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_bools(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bool]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                if (value.1[0]) {
        \\                    if (value.1[1]) {
        \\                        return value.0 + 2;
        \\                    }
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_tags(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bytes4]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                const expected: bytes4 = hex"aabbccdd";
        \\                if (value.1[0] == expected) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_tag_one(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bytes1]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                const expected: bytes1 = hex"aa";
        \\                if (value.1[0] == expected) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_tag_full(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bytes32]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                const expected: bytes32 = hex"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        \\                if (value.1[0] == expected) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const string_fn = try functionSlice(rendered, "decode_string");
    const bytes_fn = try functionSlice(rendered, "decode_bytes");
    const values_fn = try functionSlice(rendered, "decode_values");
    const addresses_fn = try functionSlice(rendered, "decode_addresses");
    const bools_fn = try functionSlice(rendered, "decode_bools");
    const tags_fn = try functionSlice(rendered, "decode_tags");
    const tag_one_fn = try functionSlice(rendered, "decode_tag_one");
    const tag_full_fn = try functionSlice(rendered, "decode_tag_full");
    const pair_text_fn = try functionSlice(rendered, "decode_pair_text");
    const pair_bytes_fn = try functionSlice(rendered, "decode_pair_bytes");
    const pair_values_fn = try functionSlice(rendered, "decode_pair_values");
    const pair_addresses_fn = try functionSlice(rendered, "decode_pair_addresses");
    const pair_bools_fn = try functionSlice(rendered, "decode_pair_bools");
    const pair_tags_fn = try functionSlice(rendered, "decode_pair_tags");
    const pair_tag_one_fn = try functionSlice(rendered, "decode_pair_tag_one");
    const pair_tag_full_fn = try functionSlice(rendered, "decode_pair_tag_full");

    for ([_][]const u8{ string_fn, bytes_fn }) |decode_fn| {
        try expectDynamicAbiDecodeGuardChain(decode_fn);
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "mload256"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload8"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "div"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mul"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "sub"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 3, "lt"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "gt"));
        try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    }
    try expectDynamicAbiDecodeWordGuardChain(values_fn);
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 2, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, values_fn, 1, "mload8"));
    try testing.expect(!std.mem.containsAtLeast(u8, values_fn, 1, "bitcast"));
    try expectDynamicAbiDecodeWordGuardChain(addresses_fn);
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, addresses_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(bools_fn);
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 1, "or"));
    try testing.expect(!std.mem.containsAtLeast(u8, bools_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(tags_fn);
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, tags_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(tag_one_fn);
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_one_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(tag_full_fn);
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_full_fn, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_full_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_full_fn, 1, "mload8"));
    try expectDynamicAbiDecodeGuardChain(pair_text_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "mload8"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "sub"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 2, "gt"));
    try expectMixedDynamicTupleCarrierShape(pair_text_fn);
    try expectDynamicAbiDecodeGuardChain(pair_bytes_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "mload8"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "sub"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 2, "gt"));
    try expectMixedDynamicTupleCarrierShape(pair_bytes_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_values_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_values_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_values_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_addresses_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_addresses_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_bools_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 1, "or"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_bools_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_bools_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_tags_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 2, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 4, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tags_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_tags_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_tag_one_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 2, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 4, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_one_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_tag_one_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_tag_full_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 2, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 4, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_full_fn, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_full_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_full_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_tag_full_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode u256 tuple memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_pair(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, u256), payload);
        \\        return match (decoded) {
        \\            Ok(pair) => pair.0 + pair.1,
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
    const decode_fn = try functionSlice(rendered, "decode_pair");

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_pair:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "const 0x20"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_create"));
}

test "compiler converts runtime abiDecode mixed static tuple memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_pair(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, bool), payload);
        \\        return match (decoded) {
        \\            Ok(pair) => pair.0,
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
    const decode_fn = try functionSlice(rendered, "decode_pair");

    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "or"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_create"));
}

test "compiler converts runtime abiDecode nested static tuple memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_nested(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, (bool, u256)), payload);
        \\        return match (decoded) {
        \\            Ok(pair) => pair.0 + pair.1.1,
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
    const decode_fn = try functionSlice(rendered, "decode_nested");

    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 3, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "const 0x20"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "or"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_create"));
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

test "compiler lowers requires clauses to runtime revert and erases ensures before SIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(amount: u256) -> u256
        \\        requires amount < 10
        \\        ensures result == amount
        \\    {
        \\        return amount;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    {
        const before_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
        defer if (before_ref.data != null) mlir.oraStringRefFree(before_ref);
        const before = before_ref.data[0..before_ref.length];
        try testing.expect(std.mem.containsAtLeast(u8, before, 1, "ora.requires"));
        try testing.expect(std.mem.containsAtLeast(u8, before, 1, "ora.ensures"));
    }

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.requires"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.ensures"));
}

test "refinement cleanup deduplicates requires checks implied by parameter refinement" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value >= 0
        \\        requires value <= 200
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, after, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.requires"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.assert"));
}

test "refinement cleanup deduplicates zero-address requires implied by NonZeroAddress" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Check {
        \\    pub fn approve(spender: NonZeroAddress)
        \\        requires spender != std.constants.ZERO_ADDRESS
        \\        requires std.constants.ZERO_ADDRESS != spender
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, after, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.requires"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.assert"));
}

test "source-inline refined arguments do not duplicate parameter refinement guards" {
    const source_text =
        \\contract Check {
        \\    inline fn requireOwner(owner: NonZeroAddress) {
        \\    }
        \\
        \\    pub fn refined(owner: NonZeroAddress) {
        \\        requireOwner(owner);
        \\    }
        \\
        \\    pub fn unrefined(owner: address) {
        \\        requireOwner(owner);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const before_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (before_ref.data != null) mlir.oraStringRefFree(before_ref);
    const before = before_ref.data[0..before_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, before, 1, "@requireOwner__inline__owner_refined_NonZeroAddress"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, before, "ora.refinement_guard"));
}

test "refinement cleanup keeps dominated requires checks in keep-proved mode" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value >= 0
        \\        requires value <= 200
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{
        .keep_proved_checks = true,
    });

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.count(u8, after, "cf.assert") > 1);
}

test "compiler lowers keep-proved requires checks through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value >= 0
        \\        requires value <= 200
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{
        .keep_proved_checks = true,
    });

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "sir.invalid"));
}

test "refinement cleanup preserves requires checks not implied by parameter refinement" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value <= 150
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, after, "cf.assert"));
}

test "refinement cleanup preserves assert message selector on cf.assert" {
    const source_text =
        \\contract Check {
        \\    pub fn run(flag: bool) {
        \\        assert(flag, "bad");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const selector = try compiler.hir.abi.keccakSelectorHex(testing.allocator, "bad");
    defer testing.allocator.free(selector);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "cf.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "ora.assert_selector"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, selector));
}

test "refinement cleanup tags kept refinement guards as clean runtime checks" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: MinValue<u256, 10>) {
        \\        let _: u256 = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "cf.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "ora.verification_type"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "guard"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "ora.verification_context"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "refinement_guard"));
}

test "compiler lowers runtime refinement guards to clean revert through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: MinValue<u256, 10>) {
        \\        let _: u256 = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "sir.invalid"));
}

test "compiler preserves requires source order before hazardous later clauses" {
    const source_text =
        \\contract Check {
        \\    pub fn ordered(a: u256, b: u256) -> u256
        \\        requires b != 0
        \\        requires a / b < 10
        \\    {
        \\        return a;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const rendered = after_ref.data[0..after_ref.length];

    const first_branch = std.mem.indexOf(u8, rendered, "sir.cond_br") orelse return error.TestUnexpectedResult;
    const division = std.mem.indexOf(u8, rendered, "sir.div") orelse return error.TestUnexpectedResult;
    try testing.expect(first_branch < division);
}

test "corpus guard runtime clause lowers to SIR revert" {
    var compilation = try compilePackage("ora-example/smt/verification/guard_runtime_clause.ora");
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "guard_runtime_clause.ora"));
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
        \\        requires scratch <= @cast(TokenUnit, std.constants.U256_MAX) - amount
        \\    {
        \\        @create(scratch, amount);
        \\    }
        \\
        \\    pub fn retire(amount: TokenUnit)
        \\        requires scratch >= amount
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
        \\        requires debts[owner] <= 100
        \\    {
        \\        @create(debts[owner], amount);
        \\    }
        \\
        \\    pub fn settle(from: address, amount: DebtUnit)
        \\        modifies debts[from], settled
        \\        requires amount >= 0
        \\        requires amount <= 100
        \\        requires debts[from] >= -100
        \\        requires settled <= 100
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
        \\        requires balances[owner] >= amount
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

test "OraToSIR keeps private scalar helper calls word-shaped" {
    const source_text =
        \\contract PrivateScalarHelper {
        \\    storage var value: u256;
        \\
        \\    fn dec(a: u256, b: u256) -> u256 {
        \\        return a - b;
        \\    }
        \\
        \\    pub fn run() {
        \\        value = dec(10, 3);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    var found_call = false;
    var lines = std.mem.splitScalar(u8, rendered, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "icall @dec") == null) continue;
        found_call = true;

        const eq_idx = std.mem.indexOfScalar(u8, line, '=') orelse return error.TestUnexpectedResult;
        const lhs = std.mem.trim(u8, line[0..eq_idx], " \t");
        var tokens = std.mem.tokenizeAny(u8, lhs, " \t");
        try testing.expect(tokens.next() != null);
        try testing.expect(tokens.next() == null);
    }
    try testing.expect(found_call);
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
        "ora-example/apps/defi_lending_pool_fv.ora",
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
        errdefer std.debug.print("OraToSIR residual runtime-op example failed: {s}\n", .{path});
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
