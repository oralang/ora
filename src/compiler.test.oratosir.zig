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

fn functionSlice(sir_text: []const u8, function_name: []const u8) ![]const u8 {
    const header = try std.fmt.allocPrint(testing.allocator, "fn {s}:", .{function_name});
    defer testing.allocator.free(header);
    const start = std.mem.indexOf(u8, sir_text, header) orelse return error.TestUnexpectedResult;
    const search_from = start + header.len;
    const rel_end = std.mem.indexOfPos(u8, sir_text, search_from, "\nfn ");
    const end = rel_end orelse sir_text.len;
    return sir_text[start..end];
}

fn expectOrderedNeedles(haystack: []const u8, needles: []const []const u8) !void {
    var cursor: usize = 0;
    for (needles) |needle| {
        const found = std.mem.indexOfPos(u8, haystack, cursor, needle) orelse return error.TestUnexpectedResult;
        cursor = found + needle.len;
    }
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
    const first_malloc = std.mem.indexOf(u8, fn_text, "malloc") orelse return error.TestUnexpectedResult;
    const after_malloc = fn_text[first_malloc..];
    // The dedicated mixed dynamic tuple branch allocates a 2-slot tuple carrier,
    // stores the static u256, then stores the string/bytes tail pointer.
    try expectOrderedNeedles(after_malloc, &.{ "malloc", "mstore256", "const 0x20", "add", "mstore256" });
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

test "frontend reuses storage roots for compound map assignments" {
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
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, rendered, "ora.map_store"));
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
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "v1 = mload256 v0"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "v2 = mload256 v1"));
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

    const has_max_literal =
        std.mem.containsAtLeast(u8, rendered, 1, "large_const 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF") or
        std.mem.containsAtLeast(u8, rendered, 1, "sir.const -1");
    try testing.expect(has_max_literal);
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
        "add slot_history",
        "sload",
    });
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
        "tstore",
    });

    const write_total = try functionSlice(rendered, "write_total");
    try expectOrderedNeedles(write_total, &.{
        "slot_total = const 0x0",
        lock_prefix,
        "tload",
        "revert",
        "sstore slot_total",
    });

    const touch_history = try functionSlice(rendered, "touch_history");
    try expectOrderedNeedles(touch_history, &.{
        "mstore256",
        "keccak256",
        lock_prefix,
        "add",
        "tstore",
    });

    const write_history = try functionSlice(rendered, "write_history");
    try expectOrderedNeedles(write_history, &.{
        "slot_history = const 0x1",
        "mstore256",
        "keccak256",
        lock_prefix,
        "add",
        "tload",
        "revert",
        "sstore",
    });

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "tstore"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "tload"));
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
