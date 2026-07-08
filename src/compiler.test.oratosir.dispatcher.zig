const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const h = common.h;
const compileText = common.compileText;
const renderSirTextForModule = common.renderSirTextForModule;
const compilePackage = common.compilePackage;
const expectNoResidualOraRuntimeOps = common.expectNoResidualOraRuntimeOps;
const functionSlice = common.functionSlice;
const expectOrderedNeedles = common.expectOrderedNeedles;

test "dispatcher orders state-mutating functions before read-only functions" {
    // bump() is declared between the two read-only functions but must get
    // the first chain position (transactions pay dispatch gas; eth_call'd
    // views do not). The views keep declaration order behind it.
    const source_text =
        \\contract Ordered {
        \\    storage var counter: u256;
        \\
        \\    pub fn peek() -> u256 {
        \\        return counter;
        \\    }
        \\
        \\    pub fn bump() {
        \\        counter = counter + 1;
        \\    }
        \\
        \\    pub fn peek_more() -> u256 {
        \\        return counter;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{
        "0x68110B2F", // bump() — mutating, promoted to the front
        "0x59E02DD7", // peek() — read-only, declaration order preserved
        "0xCA096932", // peek_more()
    });
}

test "dispatcher splits a hot mutating prefix ahead of the cold jump table" {
    // Two mutating functions and fourteen views: the dispatcher emits a tiny
    // hot switch (linear chain, 1-2 exact checks for paying transactions)
    // whose default falls through to the cold switch (table-routed views +
    // unknown selectors). Both switches carry the mandatory exact guards.
    var source: std.Io.Writer.Allocating = .init(testing.allocator);
    defer source.deinit();
    try source.writer.writeAll(
        \\contract HotPrefix {
        \\    storage var counter: u256;
        \\
        \\    pub fn bump() {
        \\        counter = counter + 1;
        \\    }
        \\
        \\    pub fn reset() {
        \\        counter = 0;
        \\    }
        \\
    );
    for (0..14) |i| {
        try source.writer.print(
            \\    pub fn v{d}() -> u256 {{
            \\        return {d};
            \\    }}
            \\
        , .{ i, 200 + i });
    }
    try source.writer.writeAll("}\n");

    var compilation = try compileText(source.written());
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const main_fn = try functionSlice(rendered, "main");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, main_fn, "switch selector"));
    try expectOrderedNeedles(main_fn, &.{
        "0x68110B2F", // bump() in the hot switch
        "0xD826F88F", // reset() in the hot switch
        "default => @cold_dispatch",
        "cold_dispatch {",
        "switch selector",
        "default => @revert_error",
    });
}

test "callHint cold demotion lets the hot-prefix split fire on token-shaped contracts" {
    // Six mutating functions normally defeat the split (hot set > 3).
    // Cold-hinting the rare setters shrinks the effective hot set to the
    // three real traffic carriers.
    var source: std.Io.Writer.Allocating = .init(testing.allocator);
    defer source.deinit();
    try source.writer.writeAll(
        \\contract HintedToken {
        \\    storage var total: u256;
        \\
        \\    pub fn transfer(to: address, amount: u256) -> bool {
        \\        total = total + amount;
        \\        return true;
        \\    }
        \\
        \\    pub fn transferFrom(from: address, to: address, amount: u256) -> bool {
        \\        total = total + amount;
        \\        return true;
        \\    }
        \\
        \\    pub fn approve(spender: address, amount: u256) -> bool {
        \\        total = total + amount;
        \\        return true;
        \\    }
        \\
        \\    pub fn mint(to: address, amount: u256) {
        \\        @callHint(cold);
        \\        total = total + amount;
        \\    }
        \\
        \\    pub fn burn(amount: u256) {
        \\        @callHint(cold);
        \\        total = total - amount;
        \\    }
        \\
        \\    pub fn pause() {
        \\        @callHint(cold);
        \\        total = 0;
        \\    }
        \\
    );
    for (0..12) |i| {
        try source.writer.print(
            \\    pub fn v{d}() -> u256 {{
            \\        return {d};
            \\    }}
            \\
        , .{ i, 300 + i });
    }
    try source.writer.writeAll("}\n");

    var compilation = try compileText(source.written());
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const main_fn = try functionSlice(rendered, "main");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, main_fn, "switch selector"));
    try expectOrderedNeedles(main_fn, &.{
        "0xA9059CBB", // transfer — hot switch
        "0x23B872DD", // transferFrom
        "0x095EA7B3", // approve
        "default => @cold_dispatch",
    });
}

test "callHint likely promotes an on-chain-read view ahead of mutating functions" {
    // Oracle pattern: a view read on-chain via staticcall pays dispatch gas,
    // and only the developer knows its volume — likely outranks mutability.
    const source_text =
        \\contract Oracle {
        \\    storage var value: u256;
        \\
        \\    pub fn push(v: u256) {
        \\        value = v;
        \\    }
        \\
        \\    pub fn latest() -> u256 {
        \\        @callHint(likely);
        \\        return value;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{
        "0x52BFE789", // latest() — likely-hinted view, front of the chain
        "0x959AC484", // push(uint256) — unhinted mutating, behind it
    });
}

test "explicit callHint likely set splits even with a small cold remainder" {
    // D7 discriminator: two likely fns + six others (< 12) — the mutability
    // heuristic would not split here, but explicit hints are developer-
    // asserted frequency and get the split whenever the remainder is >= 4.
    var source: std.Io.Writer.Allocating = .init(testing.allocator);
    defer source.deinit();
    try source.writer.writeAll(
        \\contract HintedSmall {
        \\    storage var x: u256;
        \\
        \\    pub fn poke(v: u256) {
        \\        @callHint(likely);
        \\        x = v;
        \\    }
        \\
        \\    pub fn peek() -> u256 {
        \\        @callHint(likely);
        \\        return x;
        \\    }
        \\
    );
    for (0..6) |i| {
        try source.writer.print(
            \\    pub fn w{d}() -> u256 {{
            \\        return {d};
            \\    }}
            \\
        , .{ i, 400 + i });
    }
    try source.writer.writeAll("}\n");

    var compilation = try compileText(source.written());
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const main_fn = try functionSlice(rendered, "main");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, main_fn, "switch selector"));
    try expectOrderedNeedles(main_fn, &.{
        "default => @cold_dispatch",
        "cold_dispatch {",
    });
}

test "callHint likely overflow caps the hot switch and leads the cold layer" {
    // Five likely fns: the hot switch takes the first three by declaration;
    // the other two keep their hint rank and head the cold switch, ahead of
    // every unhinted view.
    var source: std.Io.Writer.Allocating = .init(testing.allocator);
    defer source.deinit();
    try source.writer.writeAll(
        \\contract HintOverflow {
        \\    storage var x: u256;
        \\
    );
    for (0..5) |i| {
        try source.writer.print(
            \\    pub fn h{d}(v: u256) {{
            \\        @callHint(likely);
            \\        x = x + v + {d};
            \\    }}
            \\
        , .{ i, i });
    }
    for (0..14) |i| {
        try source.writer.print(
            \\    pub fn r{d}() -> u256 {{
            \\        return {d};
            \\    }}
            \\
        , .{ i, 600 + i });
    }
    try source.writer.writeAll("}\n");

    var compilation = try compileText(source.written());
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const main_fn = try functionSlice(rendered, "main");
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, main_fn, "switch selector"));
    const hot_end = std.mem.indexOf(u8, main_fn, "default => @cold_dispatch") orelse return error.TestUnexpectedResult;
    const hot_slice = main_fn[0..hot_end];
    // Exactly three hot cases: h0, h1, h2.
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, hot_slice, "=> @h"));
    // Overflow likely fns h3/h4 appear right after the cold switch opens,
    // before any view.
    const cold_slice = main_fn[hot_end..];
    const h3_pos = std.mem.indexOf(u8, cold_slice, "=> @h3_") orelse return error.TestUnexpectedResult;
    const h4_pos = std.mem.indexOf(u8, cold_slice, "=> @h4_") orelse return error.TestUnexpectedResult;
    const first_view_pos = std.mem.indexOf(u8, cold_slice, "=> @r0_") orelse return error.TestUnexpectedResult;
    try testing.expect(h3_pos < first_view_pos);
    try testing.expect(h4_pos < first_view_pos);
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
