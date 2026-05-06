const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;
const mlir = @import("mlir_c_api").c;
const z3_verification = @import("ora_z3_verification");

pub fn compileText(source_text: []const u8) !compiler.driver.Compilation {
    return compiler.compileSource(testing.allocator, "test.ora", source_text);
}

pub fn renderHirTextForSource(source_text: []const u8) ![]u8 {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    return hir_result.renderText(testing.allocator);
}

pub fn renderOraMlirForSource(source_text: []const u8) ![]u8 {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    return try testing.allocator.dupe(u8, module_text_ref.data[0..module_text_ref.length]);
}

pub fn renderSirTextForModule(context: mlir.MlirContext, module: mlir.MlirModule) ![]u8 {
    const sir_text_ref = mlir.oraEmitSIRText(context, module);
    defer if (sir_text_ref.data != null) mlir.oraStringRefFree(sir_text_ref);
    if (sir_text_ref.data == null) return error.TestUnexpectedResult;
    return try testing.allocator.dupe(u8, sir_text_ref.data[0..sir_text_ref.length]);
}

pub fn compilePackage(root_path: []const u8) !compiler.driver.Compilation {
    return compiler.compilePackage(testing.allocator, root_path);
}

pub fn expectOraToSirConverts(path: []const u8) !void {
    var compilation = try compilePackage(path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
}

pub fn expectNoResidualOraRuntimeOps(rendered: []const u8) !void {
    const forbidden = [_][]const u8{
        "ora.global",
        "ora.sload",
        "ora.sstore",
        "ora.tload",
        "ora.tstore",
        "ora.map_get",
        "ora.map_store",
        "ora.return",
        "ora.error.ok",
        "ora.error.err",
        "ora.error.is_error",
        "ora.error.unwrap",
        "ora.error.get_error",
        "ora.error.return",
        "ora.if",
        "ora.try_stmt",
        "ora.switch",
        "ora.yield",
        "ora.break",
        "ora.continue",
        "ora.conditional_return",
        "ora.struct_instantiate",
        "ora.struct_field_extract",
        "ora.struct_field_update",
        "ora.struct.decl",
        "ora.tuple_create",
        "ora.tuple_extract",
        "ora.abi_encode",
        "ora.external_call",
        "ora.abi_decode",
        "ora.assert",
        "ora.length",
        "ora.byte_at",
        "ora.log",
        "ora.lock",
        "ora.unlock",
        "ora.refinement_to_base",
        "ora.base_to_refinement",
    };

    for (forbidden) |needle| {
        const as_result = try std.fmt.allocPrint(testing.allocator, "= {s}", .{needle});
        defer testing.allocator.free(as_result);
        const as_stmt = try std.fmt.allocPrint(testing.allocator, "\n    {s}", .{needle});
        defer testing.allocator.free(as_stmt);
        if (std.mem.containsAtLeast(u8, rendered, 1, as_result) or std.mem.containsAtLeast(u8, rendered, 1, as_stmt)) {
            return error.TestUnexpectedResult;
        }
    }
}

pub const VerificationProbeSummary = struct {
    success: bool,
    errors_len: usize,
    diagnostics_len: usize,
    degraded: bool,
    error_kinds: []u8,

    pub fn deinit(self: *VerificationProbeSummary, allocator: std.mem.Allocator) void {
        allocator.free(self.error_kinds);
    }
};

pub fn expectVerificationProbeEquivalent(lhs: *const VerificationProbeSummary, rhs: *const VerificationProbeSummary) !void {
    try testing.expectEqual(lhs.success, rhs.success);
    try testing.expectEqual(lhs.errors_len, rhs.errors_len);
    try testing.expectEqual(lhs.diagnostics_len, rhs.diagnostics_len);
    try testing.expectEqualStrings(lhs.error_kinds, rhs.error_kinds);
    try testing.expectEqual(lhs.degraded, rhs.degraded);
}

pub fn verifyExampleWithoutDegradation(
    path: []const u8,
    function_name: ?[]const u8,
    parallel: bool,
    timeout_ms: ?u32,
) !VerificationProbeSummary {
    var compilation = try compilePackage(path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    errdefer verifier.deinit();
    verifier.parallel = parallel;
    verifier.filter_function_name = function_name;
    verifier.timeout_ms = timeout_ms;

    var result = if (parallel)
        try verifier.runVerificationPass(hir_result.module.raw_module)
    else
        try verifier.runVerificationPassPreparedSequential(hir_result.module.raw_module);
    errdefer result.deinit();
    const degraded = verifier.encoder.isDegraded();
    var kinds = std.ArrayList([]const u8){};
    defer kinds.deinit(testing.allocator);
    for (result.errors.items) |err| {
        try kinds.append(testing.allocator, @tagName(err.error_type));
    }
    std.mem.sort([]const u8, kinds.items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.lessThan);

    var builder = std.ArrayList(u8){};
    defer builder.deinit(testing.allocator);
    for (kinds.items, 0..) |kind, i| {
        if (i != 0) try builder.append(testing.allocator, ',');
        try builder.appendSlice(testing.allocator, kind);
    }

    defer result.deinit();
    verifier.deinit();
    return .{
        .success = result.success,
        .errors_len = result.errors.items.len,
        .diagnostics_len = result.diagnostics.items.len,
        .degraded = degraded,
        .error_kinds = try builder.toOwnedSlice(testing.allocator),
    };
}

pub fn verifyTextWithoutDegradation(source_text: []const u8, function_name: ?[]const u8) !VerificationProbeSummary {
    return verifyTextWithoutDegradationWithTimeout(source_text, function_name, null);
}

pub fn verifyTextWithoutDegradationWithTimeout(
    source_text: []const u8,
    function_name: ?[]const u8,
    timeout_ms: ?u32,
) !VerificationProbeSummary {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    errdefer verifier.deinit();
    verifier.filter_function_name = function_name;
    verifier.timeout_ms = timeout_ms;

    var result = try verifier.runVerificationPassPreparedSequential(hir_result.module.raw_module);
    errdefer result.deinit();
    const degraded = verifier.encoder.isDegraded();
    var kinds = std.ArrayList([]const u8){};
    defer kinds.deinit(testing.allocator);
    for (result.errors.items) |err| {
        try kinds.append(testing.allocator, @tagName(err.error_type));
    }
    std.mem.sort([]const u8, kinds.items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.lessThan);

    var builder = std.ArrayList(u8){};
    defer builder.deinit(testing.allocator);
    for (kinds.items, 0..) |kind, i| {
        if (i != 0) try builder.append(testing.allocator, ',');
        try builder.appendSlice(testing.allocator, kind);
    }

    defer result.deinit();
    verifier.deinit();
    return .{
        .success = result.success,
        .errors_len = result.errors.items.len,
        .diagnostics_len = result.diagnostics.items.len,
        .degraded = degraded,
        .error_kinds = try builder.toOwnedSlice(testing.allocator),
    };
}

pub fn firstChildNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind) ?compiler.SyntaxNode {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| if (child_node.kind() == kind) return child_node,
            .token => {},
        }
    }
    return null;
}

pub fn nthChildNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, ordinal: usize) ?compiler.SyntaxNode {
    var seen: usize = 0;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| {
                if (child_node.kind() != kind) continue;
                if (seen == ordinal) return child_node;
                seen += 1;
            },
            .token => {},
        }
    }
    return null;
}

pub fn containsNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind) bool {
    if (node.kind() == kind) return true;

    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| if (containsNodeOfKind(child_node, kind)) return true,
            .token => {},
        }
    }
    return false;
}

pub fn findVariablePatternByName(ast_file: *const compiler.ast.AstFile, statements: []const compiler.ast.StmtId, name: []const u8) ?compiler.ast.PatternId {
    for (statements) |statement_id| {
        const statement = ast_file.statement(statement_id).*;
        if (statement != .VariableDecl) continue;
        const pattern_id = statement.VariableDecl.pattern;
        const pattern = ast_file.pattern(pattern_id).*;
        if (pattern != .Name) continue;
        if (std.mem.eql(u8, pattern.Name.name, name)) return pattern_id;
    }
    return null;
}

pub fn diagnosticMessagesContain(diags: *const compiler.diagnostics.DiagnosticList, needle: []const u8) bool {
    for (diags.items.items) |diag| {
        if (std.mem.containsAtLeast(u8, diag.message, 1, needle)) return true;
    }
    return false;
}

pub fn countDiagnosticMessages(diags: *const compiler.diagnostics.DiagnosticList, needle: []const u8) usize {
    var count: usize = 0;
    for (diags.items.items) |diag| {
        if (std.mem.eql(u8, diag.message, needle)) count += 1;
    }
    return count;
}

pub const DiagnosticProbePhase = enum {
    syntax,
    resolution,
    typecheck,
};

pub fn expectDiagnosticProbeContains(source_text: []const u8, phase: DiagnosticProbePhase, needle: []const u8) !void {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    switch (phase) {
        .syntax => {
            const file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
            const diags = try compilation.db.syntaxDiagnostics(file_id);
            try testing.expect(diagnosticMessagesContain(diags, needle));
        },
        .resolution => {
            const diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
            try testing.expect(diagnosticMessagesContain(diags, needle));
        },
        .typecheck => {
            const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
            try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, needle));
        },
    }
}

pub fn containsEffectSlot(items: []const compiler.sema.EffectSlot, needle: []const u8, region: compiler.sema.Region) bool {
    for (items) |item| {
        if (item.region == region and std.mem.eql(u8, item.name, needle)) return true;
    }
    return false;
}

pub fn containsKeyedEffectSlot(items: []const compiler.sema.EffectSlot, needle: []const u8, region: compiler.sema.Region, key_path: []const compiler.sema.KeySegment) bool {
    for (items) |item| {
        if (item.region != region) continue;
        if (!std.mem.eql(u8, item.name, needle)) continue;
        const item_path = item.key_path orelse continue;
        if (item_path.len != key_path.len) continue;
        var all_match = true;
        for (item_path, key_path) |lhs, rhs| {
            if (!std.meta.eql(lhs, rhs)) {
                all_match = false;
                break;
            }
        }
        if (all_match) return true;
    }
    return false;
}

pub fn nthDescendantNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, ordinal: usize) ?compiler.SyntaxNode {
    var remaining = ordinal;
    return nthDescendantNodeOfKindInner(node, kind, &remaining);
}

pub fn nthDescendantNodeOfKindInner(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, remaining: *usize) ?compiler.SyntaxNode {
    if (node.kind() == kind) {
        if (remaining.* == 0) return node;
        remaining.* -= 1;
    }

    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| {
                if (nthDescendantNodeOfKindInner(child_node, kind, remaining)) |found| return found;
            },
            .token => {},
        }
    }
    return null;
}
