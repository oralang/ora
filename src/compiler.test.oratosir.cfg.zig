const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const mlir_cfg = common.mlir_cfg;
const z3_verification = common.z3_verification;
const h = common.h;
const compileText = common.compileText;
const createOraMlirContext = common.createOraMlirContext;
const parseOraModule = common.parseOraModule;
const printModuleTextForTest = common.printModuleTextForTest;
const parseSirDotGraph = common.parseSirDotGraph;

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

test "compiler does not mark guard-clause obligations as erased in Ora CFG overlay" {
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
    var formal_bindings = try h.collectFormalQueryBindingsForVerifier(testing.allocator, hir_result.module.raw_module);
    defer {
        formal_bindings.formal_result.deinit();
        testing.allocator.free(formal_bindings.z3_bindings);
    }
    verifier.setFormalQueryBindings(formal_bindings.z3_bindings);
    var vr = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer vr.deinit();

    try testing.expect(vr.success);
    try testing.expectEqual(@as(usize, 0), vr.proven_guard_ids.count());
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
    try testing.expect(!std.mem.containsAtLeast(u8, dot, 1, "proof=\"proven-erased\""));
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
