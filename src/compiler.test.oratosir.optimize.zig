const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const compileText = common.compileText;
const renderSirTextForModule = common.renderSirTextForModule;
const createOraMlirContext = common.createOraMlirContext;
const parseOraModule = common.parseOraModule;
const printModuleTextForTest = common.printModuleTextForTest;
const setModuleBoolAttr = common.setModuleBoolAttr;
const functionSlice = common.functionSlice;
const oraFunctionSlice = common.oraFunctionSlice;
const countSirBitcastsForSource = common.countSirBitcastsForSource;
const renderSirTextForSourceWithAttrs = common.renderSirTextForSourceWithAttrs;

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
