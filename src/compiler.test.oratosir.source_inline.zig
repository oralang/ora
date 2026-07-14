const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const compileText = common.compileText;
const renderSirTextForModule = common.renderSirTextForModule;
const expectNoResidualOraRuntimeOps = common.expectNoResidualOraRuntimeOps;
const createOraMlirContext = common.createOraMlirContext;
const parseOraModule = common.parseOraModule;
const functionSlice = common.functionSlice;

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
