pub const std = @import("std");
pub const testing = std.testing;
pub const ora_root = @import("ora_root");
pub const compiler = ora_root.compiler;
pub const mlir = @import("mlir_c_api").c;
pub const mlir_cfg = @import("mlir/cfg.zig");
pub const runtime_checks = @import("mlir/runtime_checks.zig");
pub const z3_verification = @import("ora_z3_verification");

pub const h = @import("compiler.test.helpers.zig");
pub const compileText = h.compileText;
pub const renderHirTextForSource = h.renderHirTextForSource;
pub const renderOraMlirForSource = h.renderOraMlirForSource;
pub const renderSirTextForModule = h.renderSirTextForModule;
pub const compilePackage = h.compilePackage;
pub const expectOraToSirConverts = h.expectOraToSirConverts;
pub const expectNoResidualOraRuntimeOps = h.expectNoResidualOraRuntimeOps;
pub const VerificationProbeSummary = h.VerificationProbeSummary;
pub const expectVerificationProbeEquivalent = h.expectVerificationProbeEquivalent;
pub const verifyExampleWithoutDegradation = h.verifyExampleWithoutDegradation;
pub const verifyTextWithoutDegradation = h.verifyTextWithoutDegradation;
pub const verifyTextWithoutDegradationWithTimeout = h.verifyTextWithoutDegradationWithTimeout;
pub const firstChildNodeOfKind = h.firstChildNodeOfKind;
pub const nthChildNodeOfKind = h.nthChildNodeOfKind;
pub const containsNodeOfKind = h.containsNodeOfKind;
pub const findVariablePatternByName = h.findVariablePatternByName;
pub const diagnosticMessagesContain = h.diagnosticMessagesContain;
pub const countDiagnosticMessages = h.countDiagnosticMessages;
pub const DiagnosticProbePhase = h.DiagnosticProbePhase;
pub const expectDiagnosticProbeContains = h.expectDiagnosticProbeContains;
pub const containsEffectSlot = h.containsEffectSlot;
pub const containsKeyedEffectSlot = h.containsKeyedEffectSlot;
pub const nthDescendantNodeOfKind = h.nthDescendantNodeOfKind;
pub const nthDescendantNodeOfKindInner = h.nthDescendantNodeOfKindInner;

pub fn extractSirGlobalSlotsJson(context: mlir.MlirContext, module: mlir.MlirModule) ![]u8 {
    const slots_ref = mlir.oraExtractSIRGlobalSlots(context, module);
    defer if (slots_ref.data != null) mlir.oraStringRefFree(slots_ref);
    if (slots_ref.data == null or slots_ref.length == 0) return error.TestUnexpectedResult;
    return try testing.allocator.dupe(u8, slots_ref.data[0..slots_ref.length]);
}

pub fn expectGlobalSlot(slots_json: []const u8, name: []const u8, expected: i128) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, slots_json, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    const actual = parsed.value.object.get(name) orelse return error.TestUnexpectedResult;
    try testing.expect(actual == .integer);
    try testing.expectEqual(expected, actual.integer);
}

pub fn createOraMlirContext() mlir.MlirContext {
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

pub fn parseOraModule(ctx: mlir.MlirContext, text: []const u8) !mlir.MlirModule {
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    if (mlir.oraModuleIsNull(module)) return error.TestUnexpectedResult;
    return module;
}

pub fn printModuleTextForTest(module: mlir.MlirModule) ![]u8 {
    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    if (module_text_ref.data == null or module_text_ref.length == 0) return error.TestUnexpectedResult;
    return testing.allocator.dupe(u8, module_text_ref.data[0..module_text_ref.length]);
}

pub fn setModuleBoolAttr(ctx: mlir.MlirContext, module: mlir.MlirModule, name: []const u8) void {
    const attr = mlir.oraBoolAttrCreate(ctx, true);
    mlir.oraOperationSetAttributeByName(
        mlir.oraModuleGetOperation(module),
        mlir.oraStringRefCreate(name.ptr, name.len),
        attr,
    );
}

pub fn functionSlice(sir_text: []const u8, function_name: []const u8) ![]const u8 {
    const header = try std.fmt.allocPrint(testing.allocator, "fn {s}:", .{function_name});
    defer testing.allocator.free(header);
    const start = std.mem.indexOf(u8, sir_text, header) orelse return error.TestUnexpectedResult;
    const search_from = start + header.len;
    const rel_end = std.mem.indexOfPos(u8, sir_text, search_from, "\nfn ");
    const end = rel_end orelse sir_text.len;
    return sir_text[start..end];
}

pub fn oraFunctionSlice(ora_text: []const u8, function_name: []const u8) ![]const u8 {
    const header = try std.fmt.allocPrint(testing.allocator, "func.func @{s}", .{function_name});
    defer testing.allocator.free(header);
    const start = std.mem.indexOf(u8, ora_text, header) orelse return error.TestUnexpectedResult;
    const search_from = start + header.len;
    const rel_end = std.mem.indexOfPos(u8, ora_text, search_from, "func.func @");
    const end = rel_end orelse ora_text.len;
    return ora_text[start..end];
}

pub fn countSirBitcastsForSource(
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

pub fn renderSirTextForSourceWithAttrs(
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

pub fn expectOrderedNeedles(haystack: []const u8, needles: []const []const u8) !void {
    var cursor: usize = 0;
    for (needles) |needle| {
        const found = std.mem.indexOfPos(u8, haystack, cursor, needle) orelse return error.TestUnexpectedResult;
        cursor = found + needle.len;
    }
}

pub const SirDotNode = struct {
    id: []const u8,
    term: []const u8,
    entry: bool = false,
    is_unreachable: bool = false,
    revert: bool = false,
};

pub const SirDotEdge = struct {
    src: []const u8,
    dst: []const u8,
    label: ?[]const u8 = null,
    backedge: bool = false,
};

pub const SirDotGraph = struct {
    nodes: []SirDotNode,
    edges: []SirDotEdge,

    pub fn deinit(self: SirDotGraph, allocator: std.mem.Allocator) void {
        allocator.free(self.nodes);
        allocator.free(self.edges);
    }

    pub fn nodeIndex(self: SirDotGraph, id: []const u8) ?usize {
        for (self.nodes, 0..) |node, index| {
            if (std.mem.eql(u8, node.id, id)) return index;
        }
        return null;
    }

    pub fn countTerm(self: SirDotGraph, term: []const u8) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (std.mem.eql(u8, node.term, term)) count += 1;
        }
        return count;
    }

    pub fn countEntryNodes(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (node.entry) count += 1;
        }
        return count;
    }

    pub fn countUnreachableNodes(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (node.is_unreachable) count += 1;
        }
        return count;
    }

    pub fn countRevertNodes(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            if (node.revert) count += 1;
        }
        return count;
    }

    pub fn countBackedges(self: SirDotGraph) usize {
        var count: usize = 0;
        for (self.edges) |edge| {
            if (edge.backedge) count += 1;
        }
        return count;
    }

    pub fn expectAllEdgesReferToKnownNodes(self: SirDotGraph) !void {
        for (self.edges) |edge| {
            try testing.expect(self.nodeIndex(edge.src) != null);
            try testing.expect(self.nodeIndex(edge.dst) != null);
        }
    }

    pub fn expectRevertNodesHaveNoSuccessors(self: SirDotGraph) !void {
        for (self.nodes) |node| {
            if (!node.revert) continue;
            for (self.edges) |edge| {
                try testing.expect(!std.mem.eql(u8, edge.src, node.id));
            }
        }
    }

    pub fn expectEveryCondBrHasTrueFalseEdges(self: SirDotGraph) !void {
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

pub fn quotedDotAttr(line: []const u8, name: []const u8) ?[]const u8 {
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

pub fn nodeTermFromLine(line: []const u8) []const u8 {
    const term_start = std.mem.indexOf(u8, line, "term=") orelse return "";
    const value_start = term_start + "term=".len;
    const rest = line[value_start..];
    if (std.mem.indexOfScalar(u8, rest, '\\')) |end| return rest[0..end];
    if (std.mem.indexOfScalar(u8, rest, '"')) |end| return rest[0..end];
    return rest;
}

pub fn parseSirDotGraph(allocator: std.mem.Allocator, dot: []const u8) !SirDotGraph {
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

// Runtime abiDecode structural helpers pin the guard shape for each decoded
// category: static head-only values, dynamic byte-padded values (string/bytes
// and the current mixed tuple), and dynamic word-only arrays such as slice[u256].
pub fn expectStaticAbiDecodeGuardBeforePayloadLoad(fn_text: []const u8) !void {
    const branch_index = std.mem.indexOf(u8, fn_text, " ? @") orelse return error.TestUnexpectedResult;
    const before_guard = fn_text[0..branch_index];
    const after_guard = fn_text[branch_index..];
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, before_guard, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 1, "mload256"));
}

pub fn expectDynamicAbiDecodeGuardChain(fn_text: []const u8) !void {
    const branch_index = std.mem.indexOf(u8, fn_text, " ? @") orelse return error.TestUnexpectedResult;
    const before_guard = fn_text[0..branch_index];
    const after_guard = fn_text[branch_index..];
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, before_guard, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 2, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 1, "mload8"));
}

pub fn expectDynamicAbiDecodeWordGuardChain(fn_text: []const u8) !void {
    const branch_index = std.mem.indexOf(u8, fn_text, " ? @") orelse return error.TestUnexpectedResult;
    const before_guard = fn_text[0..branch_index];
    const after_guard = fn_text[branch_index..];
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, before_guard, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, after_guard, 2, "mload256"));
}

pub fn expectMixedDynamicTupleCarrierShape(fn_text: []const u8) !void {
    // The dedicated mixed dynamic tuple branch allocates a 2-slot tuple carrier,
    // stores the static u256, then stores the string/bytes tail pointer.
    try expectOrderedNeedles(fn_text, &.{ "const 0x40", "mload256", "malloc", "mstore256", "add", "mstore256" });
    try testing.expect(std.mem.containsAtLeast(u8, fn_text, 1, "const 0x20"));
    try testing.expect(std.mem.containsAtLeast(u8, fn_text, 2, "malloc"));
}

pub fn firstGuardIdFromModuleText(text: []const u8) ![]const u8 {
    return nthGuardIdFromModuleText(text, 0);
}

pub fn nthGuardIdFromModuleText(text: []const u8, expected_index: usize) ![]const u8 {
    const marker = "ora.guard_id = \"";
    var offset: usize = 0;
    var index: usize = 0;
    while (std.mem.indexOf(u8, text[offset..], marker)) |relative_marker| {
        const start = offset + relative_marker + marker.len;
        const end = start + (std.mem.indexOfScalar(u8, text[start..], '"') orelse return error.TestUnexpectedResult);
        if (index == expected_index) return text[start..end];
        index += 1;
        offset = end + 1;
    }
    return error.TestUnexpectedResult;
}

pub fn setAllRefinementGuardIds(ctx: mlir.MlirContext, module: mlir.MlirModule, guard_id: []const u8) usize {
    var count: usize = 0;
    setAllRefinementGuardIdsInOp(ctx, mlir.oraModuleGetOperation(module), guard_id, &count);
    return count;
}

pub fn setAllRefinementGuardIdsInOp(
    ctx: mlir.MlirContext,
    op: mlir.MlirOperation,
    guard_id: []const u8,
    count: *usize,
) void {
    if (mlir.oraOperationIsNull(op)) return;

    const name_ref = mlir.oraOperationGetName(op);
    if (name_ref.data != null and std.mem.eql(u8, name_ref.data[0..name_ref.length], "ora.refinement_guard")) {
        mlir.oraOperationSetAttributeByName(
            op,
            mlir.oraStringRefCreate("ora.guard_id".ptr, "ora.guard_id".len),
            mlir.oraStringAttrCreate(ctx, mlir.oraStringRefCreate(guard_id.ptr, guard_id.len)),
        );
        count.* += 1;
    }

    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        if (mlir.oraRegionIsNull(region)) continue;
        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                setAllRefinementGuardIdsInOp(ctx, child, guard_id, count);
            }
        }
    }
}
