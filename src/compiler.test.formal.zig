const std = @import("std");
const testing = std.testing;
const mlir = @import("mlir_c_api").c;
const z3_verification = @import("ora_z3_verification");

const obligation = @import("formal/obligation.zig");
const obligation_crosscheck = @import("formal/obligation_crosscheck.zig");
const obligation_dump = @import("formal/obligation_dump.zig");
const obligation_from_mlir = @import("formal/obligation_from_mlir.zig");
const obligation_from_z3 = @import("formal/obligation_from_z3.zig");
const obligation_to_lean = @import("formal/obligation_to_lean.zig");
const obligation_to_z3 = @import("formal/obligation_to_z3.zig");

const test_helpers = @import("compiler.test.helpers.zig");
const compilePackage = test_helpers.compilePackage;

const ORA_BINARY_REL = "zig-out/bin/ora";

const MlirContextHandle = struct {
    ctx: mlir.MlirContext,
};

fn createContext() MlirContextHandle {
    const ctx = mlir.oraContextCreate();
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
    mlir.oraContextLoadSIRDialect(ctx);
    _ = mlir.oraDialectRegister(ctx);
    return .{ .ctx = ctx };
}

fn destroyContext(handle: MlirContextHandle) void {
    mlir.oraContextDestroy(handle.ctx);
}

fn parseModule(ctx: mlir.MlirContext, text: []const u8) !mlir.MlirModule {
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    if (mlir.oraModuleIsNull(module)) return error.MlirParseFailed;
    return module;
}

fn findFirstOpByName(module: mlir.MlirModule, name: []const u8) ?mlir.MlirOperation {
    if (mlir.oraModuleIsNull(module)) return null;
    return findFirstOpByNameInOp(mlir.oraModuleGetOperation(module), name);
}

fn findFirstOpByNameInOp(op: mlir.MlirOperation, expected: []const u8) ?mlir.MlirOperation {
    if (mlir.oraOperationIsNull(op)) return null;

    const name_ref = mlir.oraOperationGetName(op);
    if (name_ref.data != null and std.mem.eql(u8, name_ref.data[0..name_ref.length], expected)) return op;

    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        if (mlir.oraRegionIsNull(region)) continue;

        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                if (findFirstOpByNameInOp(child, expected)) |found| return found;
            }
        }
    }

    return null;
}

fn strRef(text: []const u8) mlir.MlirStringRef {
    return .{ .data = text.ptr, .length = text.len };
}

fn countLogical(set: obligation.ObligationSet, role: obligation.LogicalRole) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .logical and item.kind.logical.role == role) count += 1;
    }
    return count;
}

fn countRuntimeGuards(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .runtime_guard) count += 1;
    }
    return count;
}

fn countAssumption(set: obligation.ObligationSet, kind: obligation.AssumptionKind) usize {
    var count: usize = 0;
    for (set.assumptions) |item| {
        if (item.kind == kind) count += 1;
    }
    return count;
}

fn countResource(set: obligation.ObligationSet, op: obligation.ResourceOperation, property: obligation.ResourceProperty) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .resource and item.kind.resource.op == op and item.kind.resource.property == property) count += 1;
    }
    return count;
}

fn countQuantifier(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .quantifier) count += 1;
    }
    return count;
}

fn countEffectFrame(set: obligation.ObligationSet, relation: obligation.EffectFrameRelation) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .effect_frame and item.kind.effect_frame.relation == relation) count += 1;
    }
    return count;
}

fn countArithmeticSafety(set: obligation.ObligationSet, safety: obligation.ArithmeticSafetyKind) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .logical and
            item.kind.logical.role == .arithmetic_safety and
            item.kind.logical.arithmetic_safety != null and
            item.kind.logical.arithmetic_safety.? == safety)
        {
            count += 1;
        }
    }
    return count;
}

fn countQuery(set: obligation.ObligationSet, kind: obligation.VerificationQueryKind) usize {
    var count: usize = 0;
    for (set.queries) |item| {
        if (item.kind == kind) count += 1;
    }
    return count;
}

fn countQueryRole(set: obligation.ObligationSet, role: obligation.LogicalRole) usize {
    var count: usize = 0;
    for (set.queries) |item| {
        if (item.logical_role != null and item.logical_role.? == role) count += 1;
    }
    return count;
}

fn expectParameterKey(key: obligation.PlaceKey, expected: u32) !void {
    try testing.expect(key == .parameter);
    try testing.expectEqual(expected, key.parameter);
}

fn expectPlaceRoot(place: obligation.PlaceRef, expected_root: []const u8, expected_region: obligation.RegionRef) !void {
    try testing.expectEqualStrings(expected_root, place.root);
    try testing.expectEqual(expected_region, place.region);
}

fn expectLogicalTerm(set: obligation.ObligationSet, role: obligation.LogicalRole) !obligation.TermId {
    for (set.obligations) |item| {
        if (item.kind != .logical or item.kind.logical.role != role) continue;
        if (item.kind.logical.formula != .term) return error.TestUnexpectedResult;
        return item.kind.logical.formula.term;
    }
    return error.TestUnexpectedResult;
}

fn expectQuantifiedTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_quantifier: obligation.Quantifier,
    expected_name: []const u8,
) !obligation.QuantifiedTerm {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .quantified);
    const quantified = set.terms[term_id].quantified;
    try testing.expectEqual(expected_quantifier, quantified.quantifier);
    try testing.expectEqualStrings(expected_name, quantified.binder.name);
    return quantified;
}

fn expectBinaryTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_op: obligation.BinaryOp,
) !obligation.BinaryTerm {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .binary);
    const binary = set.terms[term_id].binary;
    try testing.expectEqual(expected_op, binary.op);
    return binary;
}

fn expectFreeVarTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_file_id: u32,
    expected_pattern_id: u32,
    expected_name: []const u8,
) !void {
    _ = try expectFreeVarTermRef(set, term_id, expected_file_id, expected_pattern_id, expected_name);
}

fn expectFreeVarTermRef(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_file_id: u32,
    expected_pattern_id: u32,
    expected_name: []const u8,
) !obligation.FreeVarRef {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .variable);
    const variable = set.terms[term_id].variable;
    try testing.expect(variable == .free);
    try testing.expectEqual(expected_file_id, variable.free.id.file_id);
    try testing.expectEqual(expected_pattern_id, variable.free.id.pattern_id);
    try testing.expectEqualStrings(expected_name, variable.free.name);
    return variable.free;
}

fn expectTypeRefSpelling(ty: ?obligation.TypeRef, expected: []const u8) !void {
    const actual = ty orelse return error.TestUnexpectedResult;
    try testing.expect(actual == .spelling);
    try testing.expectEqualStrings(expected, actual.spelling);
}

fn expectBoundVarTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_index: u32,
    expected_name: []const u8,
) !void {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .variable);
    const variable = set.terms[term_id].variable;
    try testing.expect(variable == .bound);
    try testing.expectEqual(expected_index, variable.bound.index);
    try testing.expectEqualStrings(expected_name, variable.bound.name);
}

fn expectSummaryMatchesZ3PreparedQueries(
    formal_summary: obligation.VerificationQuerySummary,
    z3_summary: z3_verification.PreparedQuerySummary,
) !void {
    const formal_total =
        @as(u64, formal_summary.base) +
        @as(u64, formal_summary.obligation) +
        @as(u64, formal_summary.loop_invariant_step) +
        @as(u64, formal_summary.loop_body_safety) +
        @as(u64, formal_summary.loop_invariant_post) +
        @as(u64, formal_summary.guard_satisfy) +
        @as(u64, formal_summary.guard_violate);
    try testing.expectEqual(formal_total, z3_summary.total);
    try testing.expectEqual(@as(u64, formal_summary.base), z3_summary.base);
    try testing.expectEqual(@as(u64, formal_summary.obligation), z3_summary.obligation);
    try testing.expectEqual(@as(u64, formal_summary.loop_invariant_step), z3_summary.loop_invariant_step);
    try testing.expectEqual(@as(u64, formal_summary.loop_body_safety), z3_summary.loop_body_safety);
    try testing.expectEqual(@as(u64, formal_summary.loop_invariant_post), z3_summary.loop_invariant_post);
    try testing.expectEqual(@as(u64, formal_summary.guard_satisfy), z3_summary.guard_satisfy);
    try testing.expectEqual(@as(u64, formal_summary.guard_violate), z3_summary.guard_violate);
}

fn emitLeanToOwnedString(allocator: std.mem.Allocator, set: obligation.ObligationSet) ![]u8 {
    var buffer = std.Io.Writer.Allocating.init(allocator);
    errdefer buffer.deinit();
    try obligation_to_lean.writeModule(&buffer.writer, set, .{});
    return try buffer.toOwnedSlice();
}

fn dumpManifestToOwnedString(allocator: std.mem.Allocator, set: obligation.ObligationSet) ![]u8 {
    var buffer = std.Io.Writer.Allocating.init(allocator);
    errdefer buffer.deinit();
    try obligation_dump.writeJsonLines(&buffer.writer, set);
    return try buffer.toOwnedSlice();
}

fn collectPackageObligations(allocator: std.mem.Allocator, path: []const u8) !obligation_from_mlir.CollectResult {
    var compilation = try compilePackage(path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    return try obligation_from_mlir.collect(allocator, hir_result.module.raw_module, .{});
}

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

fn runProcess(allocator: std.mem.Allocator, argv: []const []const u8) !std.process.RunResult {
    var io_thread = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
    });
    defer io_thread.deinit();

    return std.process.run(allocator, io_thread.io(), .{
        .argv = argv,
        .stdout_limit = std.Io.Limit.limited(1024 * 1024),
        .stderr_limit = std.Io.Limit.limited(1024 * 1024),
    });
}

fn leanPathEnvArgForTest(allocator: std.mem.Allocator) ![]const u8 {
    const inherited_path = if (std.c.getenv("PATH")) |path| std.mem.span(path) else "/usr/bin:/bin:/opt/homebrew/bin";
    if (std.c.getenv("HOME")) |home| {
        const home_slice = std.mem.span(home);
        return try std.fmt.allocPrint(allocator, "PATH={s}/.elan/bin:{s}", .{ home_slice, inherited_path });
    }
    return try std.fmt.allocPrint(allocator, "PATH={s}", .{inherited_path});
}

fn runOraWithForcedUnknown(
    allocator: std.mem.Allocator,
    forced_query: []const u8,
    args: []const []const u8,
) !std.process.RunResult {
    const forced_unknown_arg = try std.fmt.allocPrint(allocator, "ORA_Z3_TEST_FORCE_UNKNOWN_QUERY={s}", .{forced_query});
    defer allocator.free(forced_unknown_arg);
    const path_arg = try leanPathEnvArgForTest(allocator);
    defer allocator.free(path_arg);

    var argv = try allocator.alloc([]const u8, args.len + 3);
    defer allocator.free(argv);
    argv[0] = "/usr/bin/env";
    argv[1] = forced_unknown_arg;
    argv[2] = path_arg;
    @memcpy(argv[3..], args);
    return runProcess(allocator, argv);
}

fn runLeanFileForTest(allocator: std.mem.Allocator, path_from_repo_root: []const u8) !std.process.RunResult {
    const path_arg = try leanPathEnvArgForTest(allocator);
    defer allocator.free(path_arg);
    const lean_path = if (std.fs.path.isAbsolute(path_from_repo_root))
        try allocator.dupe(u8, path_from_repo_root)
    else
        try std.fs.path.join(allocator, &.{ "..", path_from_repo_root });
    defer allocator.free(lean_path);

    var io_thread = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
    });
    defer io_thread.deinit();

    const argv = [_][]const u8{ "/usr/bin/env", path_arg, "lake", "env", "lean", lean_path };
    return std.process.run(allocator, io_thread.io(), .{
        .argv = &argv,
        .cwd = .{ .path = "formal" },
        .stdout_limit = std.Io.Limit.limited(1024 * 1024),
        .stderr_limit = std.Io.Limit.limited(1024 * 1024),
    });
}

fn expectExited(result: std.process.RunResult, expected: u8) !void {
    switch (result.term) {
        .exited => |code| {
            if (code != expected) {
                std.debug.print("process stdout:\n{s}\nprocess stderr:\n{s}\n", .{ result.stdout, result.stderr });
            }
            try testing.expectEqual(expected, code);
        },
        else => return error.TestUnexpectedResult,
    }
}

fn readFileAllocForTest(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        path,
        allocator,
        std.Io.Limit.limited(1024 * 1024),
    );
}

fn leanProofGeneratedNamespaceForTest(allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    const stem = std.fs.path.stem(file_path);
    var component_out = std.Io.Writer.Allocating.init(allocator);
    defer component_out.deinit();
    const writer = &component_out.writer;

    try writer.writeAll("Source_");
    for (stem) |byte| {
        if (std.ascii.isAlphanumeric(byte) or byte == '_') {
            try writer.writeByte(byte);
        } else {
            try writer.writeByte('_');
        }
    }

    const hash = std.hash.Wyhash.hash(0, file_path);
    return try std.fmt.allocPrint(
        allocator,
        "Ora.Generated.Obligations.{s}_{x}",
        .{ component_out.written(), hash },
    );
}

fn moduleSuffixFromTmp(allocator: std.mem.Allocator, tmp: std.testing.TmpDir) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    try out.writer.writeAll("Run_");
    for (tmp.sub_path) |byte| {
        if (std.ascii.isAlphanumeric(byte) or byte == '_') {
            try out.writer.writeByte(byte);
        } else {
            try out.writer.writeByte('_');
        }
    }
    return try out.toOwnedSlice();
}

fn findEnsuresQuery(set: obligation.ObligationSet) !obligation.VerificationQuery {
    for (set.queries) |query| {
        if (query.kind == .obligation and query.logical_role != null and query.logical_role.? == .ensures) return query;
    }
    return error.TestUnexpectedResult;
}

fn writeJsonStringForTest(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '\\' => try writer.writeAll("\\\\"),
            '"' => try writer.writeAll("\\\""),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

fn writeIdArrayForTest(writer: anytype, ids: []const obligation.Id) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeProofManifestForTest(
    allocator: std.mem.Allocator,
    path: []const u8,
    module_name: []const u8,
    theorem_name: []const u8,
    proof_path: []const u8,
    query: obligation.VerificationQuery,
) !void {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\n  \"schema_version\": 1,\n  \"proofs\": [\n    {\n      \"query_id\": ");
    try writer.print("{d}", .{query.id});
    try writer.writeAll(",\n      \"obligation_ids\": ");
    try writeIdArrayForTest(writer, query.obligation_ids);
    try writer.writeAll(",\n      \"assumption_ids\": ");
    try writeIdArrayForTest(writer, query.assumption_ids);
    try writer.writeAll(",\n      \"module_name\": ");
    try writeJsonStringForTest(writer, module_name);
    try writer.writeAll(",\n      \"theorem_name\": ");
    try writeJsonStringForTest(writer, theorem_name);
    try writer.writeAll(",\n      \"path\": ");
    try writeJsonStringForTest(writer, proof_path);
    try writer.writeAll(",\n      \"content_sha256\": null\n    }\n  ]\n}\n");

    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = out.written() });
}

fn proofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    use_sorry: bool,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.TestUnexpectedResult;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.TestUnexpectedResult) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.TestUnexpectedResult;

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.writeAll("namespace ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');

    try writer.print("theorem discharge : emittedQuery_{d} := by\n", .{query_id});
    if (use_sorry) {
        try writer.writeAll("  sorry\n\n");
    } else {
        try writer.writeAll("  unfold ");
        try writer.print("emittedQuery_{d}", .{query_id});
        try writer.writeAll(
            \\ obligationFollowsFromAssumptions
            \\  constructor
            \\  · refine ⟨Env.empty.setFree { file_id := 0, pattern_id := 0 } (.u256 (BitVec.ofNat 256 0)), ?_⟩
            \\    have hId :
            \\        (({ file_id := 0, pattern_id := 0 } : FreeVarId) ==
            \\          { file_id := 0, pattern_id := 0 }) = true := by
            \\      decide
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      Env.setFree,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      hId
            \\    ]
            \\  · intro env hAssumptions
            \\    intro x
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      Env.pushBound,
            \\      Env.lookupVar,
            \\      Env.lookupBound,
            \\      Value.eqProp?,
            \\      BinderRef.isU256,
            \\      BoundVarRef.isU256,
            \\      TyRef.isU256
            \\    ]
            \\    intro i hi
            \\    exact U256.ult_implies_ule i x hi
            \\
            \\
        );
    }
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

test "B3 lean proofs unblock source-level unknown without erasing runtime guard" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B3LeanProofGate {
        \\    pub fn bounded(x: u256) -> bool
        \\        requires x == x
        \\        ensures (forall i: u256 where i < x => i <= x)
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn guarded(y: u256) -> bool
        \\        guard y > 0
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b3_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b3_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const sorry_out = try pathFromTmpAlloc(allocator, tmp, "sorry");
    defer allocator.free(sorry_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
    });
    const obligations_source = obligations_source_out.written();

    const expected_query_text = try std.fmt.allocPrint(allocator, "query: emittedQuery_{d}", .{ensures_query.id});
    defer allocator.free(expected_query_text);
    const expected_theorem_text = try std.fmt.allocPrint(allocator, "theorem discharge_q{d}", .{ensures_query.id});
    defer allocator.free(expected_theorem_text);
    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{ensures_query.id});
    defer allocator.free(expected_query_def);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "could not prove ensures") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "could not prove ensures"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for Z3 UNKNOWN obligations"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, expected_query_text));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, expected_theorem_text));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, generated_namespace));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "--lean-proofs <proofs.json>"));
    }
    const fail_report_path = try pathFromTmpAlloc(allocator, tmp, "fail/b3_lean_gate.smt.report.json");
    defer allocator.free(fail_report_path);
    const fail_report = try readFileAllocForTest(allocator, fail_report_path);
    defer allocator.free(fail_report);
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"status\": \"UNKNOWN\""));
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"vacuity_unknown\": false"));
    const fail_lean_obligations_path = try pathFromTmpAlloc(allocator, tmp, "fail/b3_lean_gate.lean.obligations.lean");
    defer allocator.free(fail_lean_obligations_path);
    const fail_lean_obligations = try readFileAllocForTest(allocator, fail_lean_obligations_path);
    defer allocator.free(fail_lean_obligations);
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, generated_namespace));
    {
        const result = try runLeanFileForTest(allocator, fail_lean_obligations_path);
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b3_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B3Fixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B3Fixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B3Fixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const sorry_module = try std.fmt.allocPrint(allocator, "Ora.B3Fixture.{s}.Sorry", .{module_suffix});
    defer allocator.free(sorry_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const sorry_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{sorry_module});
    defer allocator.free(sorry_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const sorry_proof_path = try std.fmt.allocPrint(allocator, "{s}/Sorry.lean", .{fixture_dir});
    defer allocator.free(sorry_proof_path);

    const valid_proof = try proofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    const sorry_proof = try proofModuleFromGeneratedObligations(allocator, obligations_source, sorry_module, ensures_query.id, true);
    defer allocator.free(sorry_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = sorry_proof_path, .data = sorry_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    const sorry_manifest = try pathFromTmpAlloc(allocator, tmp, "sorry-proofs.json");
    defer allocator.free(sorry_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);
    try writeProofManifestForTest(allocator, sorry_manifest, sorry_module, sorry_theorem, sorry_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b3_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b3_lean_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b3_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"proof_count\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"axioms\""));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));

    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "fn guarded:"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "c0 = const 0x0"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "gt v0 c0"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "revert 0x0 0x0"));
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);

    {
        const result = try runProcess(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=sir-text,bytecode",
            "--out-dir",
            reference_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b3_lean_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b3_lean_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            sorry_out,
            "--lean-proofs",
            sorry_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof gate failed") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof gate failed"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b3_lean_gate.hex", .{}));
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b3_lean_gate.lean.proof.json", .{}));
}

test "B5 unknown diagnostic reports Lean projection coverage when no recipe is available" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const out_dir = try pathFromTmpAlloc(allocator, tmp, "b5-out");
    defer allocator.free(out_dir);

    const result = try runOraWithForcedUnknown(allocator, "obligation:3", &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report",
        "--out-dir",
        out_dir,
        "ora-example/formal/obligation_report_logical.ora",
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe unavailable for some Z3 UNKNOWN obligations"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "unmatched UNKNOWN prepared rows: 1"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "formula projection: term="));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "origin_value="));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "term_ratio_basis_points="));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "no matching Lean-dischargeable obligation query"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for Z3 UNKNOWN obligations"));
}

test "formal query vocabulary matches Z3 prepared query vocabulary" {
    inline for (std.meta.fields(z3_verification.QueryKind)) |field| {
        const z3_kind: z3_verification.QueryKind = @enumFromInt(field.value);
        const label = z3_verification.formalQueryKindLabel(z3_kind);
        const formal_kind = std.meta.stringToEnum(obligation.VerificationQueryKind, label) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(label, @tagName(formal_kind));
    }

    inline for (std.meta.fields(obligation.VerificationQueryKind)) |field| {
        var found = false;
        inline for (std.meta.fields(z3_verification.QueryKind)) |z3_field| {
            const z3_kind: z3_verification.QueryKind = @enumFromInt(z3_field.value);
            found = found or std.mem.eql(u8, field.name, z3_verification.formalQueryKindLabel(z3_kind));
        }
        try testing.expect(found);
    }
}

test "formal query fragments and solver logic match Z3 vocabulary" {
    inline for (std.meta.fields(z3_verification.QueryFragment)) |field| {
        const z3_fragment: z3_verification.QueryFragment = @enumFromInt(field.value);
        const label = z3_verification.formalQueryFragmentLabel(z3_fragment);
        const formal_fragment = std.meta.stringToEnum(obligation.VerificationQueryFragment, label) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(label, @tagName(formal_fragment));
    }

    inline for (std.meta.fields(z3_verification.QuerySolverLogic)) |field| {
        const z3_logic: z3_verification.QuerySolverLogic = @enumFromInt(field.value);
        const label = z3_verification.formalQuerySolverLogicLabel(z3_logic);
        const formal_logic = std.meta.stringToEnum(obligation.VerificationSolverLogic, label) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(label, @tagName(formal_logic));
    }
}

test "formal logical and assumption roles match Z3 annotation vocabulary" {
    inline for (std.meta.fields(z3_verification.AnnotationKind)) |field| {
        const annotation: z3_verification.AnnotationKind = @enumFromInt(field.value);
        if (z3_verification.formalLogicalRoleLabel(annotation)) |label| {
            const role = std.meta.stringToEnum(obligation.LogicalRole, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(role));
        }
        if (z3_verification.formalAssumptionKindLabel(annotation)) |label| {
            const kind = std.meta.stringToEnum(obligation.AssumptionKind, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(kind));
        }
    }
}

test "formal imported pending obligation roles match Z3 source vocabulary" {
    inline for (std.meta.fields(z3_verification.PendingObligationSourceKind)) |field| {
        const source: z3_verification.PendingObligationSourceKind = @enumFromInt(field.value);
        if (z3_verification.formalPendingObligationRoleLabel(source)) |label| {
            const role = std.meta.stringToEnum(obligation.LogicalRole, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(role));
        }
    }
}

test "formal pending constraint assumptions match Z3 source vocabulary" {
    inline for (std.meta.fields(z3_verification.PendingConstraintSourceKind)) |field| {
        const source: z3_verification.PendingConstraintSourceKind = @enumFromInt(field.value);
        if (z3_verification.formalPendingConstraintAssumptionKindLabel(source)) |label| {
            const kind = std.meta.stringToEnum(obligation.AssumptionKind, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(kind));
        }
    }
}

test "formal MLIR manifest query summary matches Z3 prepared query summary" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var formal_result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer formal_result.deinit();

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_summary = try pass.collectPreparedQuerySummary(module);

    try expectSummaryMatchesZ3PreparedQueries(
        obligation.VerificationQuerySummary.fromQueries(formal_result.set.queries),
        z3_summary,
    );
}

test "formal Z3 prepared query manifest maps into canonical queries" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var z3_manifest = try pass.collectPreparedQueryManifest(module);
    defer z3_manifest.deinit();

    var formal_result = try obligation_from_z3.collectPreparedQueries(testing.allocator, z3_manifest.rows);
    defer formal_result.deinit();

    const summary = obligation.VerificationQuerySummary.fromQueries(formal_result.set.queries);
    try testing.expectEqual(@as(u32, 1), summary.base);
    try testing.expectEqual(@as(u32, 2), summary.obligation);
    try testing.expectEqual(@as(u32, 1), summary.guard_satisfy);
    try testing.expectEqual(@as(u32, 1), summary.guard_violate);
    try testing.expectEqual(@as(usize, 1), countQueryRole(formal_result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countQueryRole(formal_result.set, .contract_invariant));

    var guarded_queries: usize = 0;
    for (formal_result.set.queries) |query| {
        try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
        try testing.expect(query.smtlib_hash != null);
        try testing.expect(query.constraint_count > 0 or query.kind == .base);
        if (query.kind == .guard_satisfy or query.kind == .guard_violate) {
            guarded_queries += 1;
            try testing.expectEqualStrings("guard:checked:flag", query.guard_id orelse return error.TestUnexpectedResult);
        }
    }
    try testing.expectEqual(@as(usize, 2), guarded_queries);
}

test "formal Z3 prepared query results overlay MLIR manifest queries" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var formal_result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer formal_result.deinit();

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var verification_result = try pass.runVerificationPass(module);
    defer verification_result.deinit();
    var report = try pass.buildSmtReport(module, "/tmp/formal_overlay.ora", &verification_result);
    defer report.deinit(testing.allocator);

    const query_manifest = report.query_manifest orelse return error.TestUnexpectedResult;
    var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal_result.set, query_manifest.rows);
    defer overlay.deinit();

    try testing.expectEqual(formal_result.set.queries.len, overlay.set.queries.len);
    try testing.expectEqual(formal_result.set.diagnostics.len, overlay.set.diagnostics.len);
    for (overlay.set.queries) |query| {
        try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
        try testing.expect(query.smtlib_hash != null);
        try testing.expect(query.result != null);
    }
}

test "formal Z3 adapter proves canonical term obligation from assumptions" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "balance", .ty = .{ .spelling = "u256" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 1 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const assumptions = [_]obligation.Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 2 },
        },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "resource_source_sufficient", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .invariant,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try adapter.checkObligation(2));
}

test "formal Z3 adapter finds counterexample for unassumed canonical obligation" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .refinement_predicate = .{ .name = "NonZero", .value = 0 } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "refinement", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .refinement,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, try adapter.checkObligation(1));
}

test "formal Z3 adapter fails closed on MLIR-origin formulas" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .origin_value = .{
                    .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
                } },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations, .queries = &queries };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectError(error.UnsupportedOriginValue, adapter.checkObligation(1));
}

test "formal Lean emitter writes manifest rows from canonical obligations" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "balance", .ty = .{ .spelling = "u256" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 1 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const assumptions = [_]obligation.Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 2 },
        },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "resource_source_sufficient", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .invariant,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "import Ora.Obligation.Semantics"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".variable (.free { id := { file_id := 0, pattern_id := 0 }, name := \"balance\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".variable (.free { id := { file_id := 0, pattern_id := 1 }, name := \"amount\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "{ id := 1, owner := \"transfer\", kind := .requires, formula := some (.term 2) }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "{ id := 2, owner := \"transfer\", kind := .logical .invariant (.term 2) }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf : emittedManifest.wf = true := by decide"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "def emittedObligation_2 : Prop :="));
}

test "formal Lean emitter writes resource rows with places" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const source_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = 0 }},
    };
    const destination_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = 1 }},
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .resource_op = .{ .op = .move, .domain = "TokenUnit" } },
            .kind = .{ .resource = .{
                .op = .move,
                .domain = "TokenUnit",
                .source = source_place,
                .destination = destination_place,
                .amount = .{ .term = 2 },
                .property = .conservation,
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".resource { op := .move, domain := \"TokenUnit\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "source := some { root := \"balances\", region := .storage, fields := [], keys := [.parameter 0] }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "destination := some { root := \"balances\", region := .storage, fields := [], keys := [.parameter 1] }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "amount := some (.term 2), property := .conservation"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf"));
}

test "formal Lean emitter writes quantifier metadata rows" {
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "bounded" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.quantified", .symbol = "bounded" } },
            .kind = .{ .quantifier = .{
                .quantifier = .forall,
                .variable = "i",
                .binder_type = .{ .spelling = "u256" },
                .binder_sort = .bit_vector,
                .fragment = .aufbv_quantifiers,
                .pattern_status = .absent,
            } },
            .artifact_policy = .diagnostic_only,
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".quantifier { quantifier := .forall_, binderName := \"i\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "binderType := .spelling \"u256\", binderSort := .bitVector"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "fragment := .aufbvQuantifiers, patternStatus := .absent, degradation := none"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf"));
}

test "formal Lean emitter writes backend fact rows" {
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .backend = .{ .component = .dispatcher, .name = "erc20" } },
            .source = .generated(),
            .phase = .sinora,
            .origin = .{ .backend_fact = .{ .component = .dispatcher, .fact = "selector_table_complete" } },
            .kind = .{ .backend_fact = .{
                .component = .dispatcher,
                .property = .preserves_selector_behavior,
            } },
            .required_backend = .lean,
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".backendFact { component := .dispatcher, property := .preservesSelectorBehavior }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf"));
}

test "formal Lean emitter writes userland proof artifact attachments" {
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const artifacts = [_]obligation.ProofArtifact{
        .{
            .id = 9,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{ .file = "proofs/ERC20/Transfer.lean" },
            .module_name = "ERC20.Transfer",
            .theorem_name = "transfer_preserves_supply",
            .path = "proofs/ERC20/Transfer.lean",
            .content_hash = 0x1234,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "def emittedProofArtifacts : List ProofArtifactRow := ["));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "kind := .userlandLean"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "moduleName := \"ERC20.Transfer\", theoremName := \"transfer_preserves_supply\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "path := some \"proofs/ERC20/Transfer.lean\", contentHash := some 4660, obligationIds := [1]"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "proofArtifacts := emittedProofArtifacts"));
}

test "formal Lean emitter fails closed on MLIR-origin formulas" {
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .origin_value = .{
                    .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
                } },
            } },
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations };

    var buffer = std.Io.Writer.Allocating.init(testing.allocator);
    defer buffer.deinit();
    try testing.expectError(error.UnsupportedOriginValue, obligation_to_lean.writeModule(&buffer.writer, set, .{}));
}

test "formal cross-check elides runtime guard only when Z3 proves and Lean exports" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const assumptions = [_]obligation.Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 2 },
        },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "guard", .ordinal = 0 } },
            .kind = .{ .runtime_guard = .{
                .guard_id = "guard:deposit:amount",
                .formula = .{ .term = 2 },
                .erasure = .may_elide_if_proven,
            } },
        },
    };
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var result = try obligation_crosscheck.crossCheckRuntimeGuard(testing.allocator, set, 2);
    defer result.deinit();

    try testing.expectEqual(@as(obligation.Id, 2), result.obligation_id);
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, result.z3_status);
    try testing.expectEqual(obligation_crosscheck.RuntimeErasureDecision.elide_runtime_check, result.runtime_erasure);
    try testing.expect(std.mem.containsAtLeast(u8, result.lean_source, 1, "theorem emitted_manifest_wf"));
    try testing.expect(std.mem.containsAtLeast(u8, result.lean_source, 1, "guard:deposit:amount"));
}

test "formal cross-check keeps runtime guard when Z3 finds a counterexample" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .refinement_predicate = .{ .name = "NonZero", .value = 0 } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "guard", .ordinal = 0 } },
            .kind = .{ .runtime_guard = .{
                .guard_id = "guard:deposit:amount",
                .formula = .{ .term = 1 },
                .erasure = .may_elide_if_proven,
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var result = try obligation_crosscheck.crossCheckRuntimeGuard(testing.allocator, set, 1);
    defer result.deinit();

    try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, result.z3_status);
    try testing.expectEqual(obligation_crosscheck.RuntimeErasureDecision.keep_runtime_check, result.runtime_erasure);
    try testing.expect(std.mem.containsAtLeast(u8, result.lean_source, 1, "emittedManifest.wf = true"));
}

test "formal cross-check keeps always-runtime guard even when proved" {
    const terms = [_]obligation.Term{
        .{ .bool_lit = true },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "critical" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "assert", .ordinal = 0 } },
            .kind = .{ .runtime_guard = .{
                .guard_id = "guard:critical:true",
                .formula = .{ .term = 0 },
                .erasure = .always_runtime,
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "critical" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var result = try obligation_crosscheck.crossCheckRuntimeGuard(testing.allocator, set, 1);
    defer result.deinit();

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, result.z3_status);
    try testing.expectEqual(obligation_crosscheck.RuntimeErasureDecision.keep_runtime_check, result.runtime_erasure);
}

test "formal obligation MLIR adapter classifies arithmetic safety producers" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @arith(%flag: i1, %x: i256, %y: i256) {
        \\    "ora.assert"(%flag) <{message = "checked addition overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked subtraction overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked multiplication overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked power overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked negation overflow"}> : (i1) -> ()
        \\    %q = arith.divui %x, %y : i256
        \\    %s = arith.shli %x, %y : i256
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 7), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 8), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 7), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .addition_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .subtraction_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .multiplication_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .power_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .negation_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .division_by_zero));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .shift_amount_bounds));
}

test "formal obligation MLIR adapter collects verification markers and assumptions" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 3), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 2), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 5), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .assume));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 2), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));
    try testing.expect(result.set.assumptions[0].owner == .function);
    try testing.expectEqualStrings("checked", result.set.assumptions[0].owner.function.name);

    const guard = result.set.obligations[2];
    try testing.expect(guard.kind == .runtime_guard);
    try testing.expectEqualStrings("guard:checked:flag", guard.kind.runtime_guard.guard_id);
    try testing.expect(guard.kind.runtime_guard.formula == .term);
}

test "formal obligation MLIR adapter binds function params by compiler id not display name" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @same_display_name(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["same", "same"],
        \\    ora.param_binding_ids = ["file:7:pattern:10", "file:7:pattern:11"]
        \\  } {
        \\    %cmp = ora.cmp "ule", %arg0, %arg1 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "same");
    try expectTypeRefSpelling(outer.binder.ty, "u256");
    const inner = try expectQuantifiedTerm(result.set, outer.body, .forall, "same");
    try expectTypeRefSpelling(inner.binder.ty, "u256");
    const body = try expectBinaryTerm(result.set, inner.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 1, "same");
    try expectBoundVarTerm(result.set, body.rhs, 0, "same");
}

test "formal obligation MLIR adapter blocks named params without compiler binding ids" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @old_metadata(%arg0: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["x"]
        \\  } {
        \\    %cmp = ora.cmp "ule", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
    try testing.expectEqual(obligation.ArtifactBlockReason.blocking_diagnostic, result.set.artifactDecision().blocked);
}

test "formal obligation MLIR adapter binds function params in inserted forall conditions" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bounded_by_requires(%arg0: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["x"],
        \\    ora.param_binding_ids = ["file:9:pattern:3"]
        \\  } {
        \\    %req = ora.cmp "ult", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%req) : (i1) -> ()
        \\    %ens = ora.cmp "ule", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%ens) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const requires_formula = result.set.assumptions[0].formula orelse return error.TestUnexpectedResult;
    try testing.expect(requires_formula == .term);
    const requires_body = try expectBinaryTerm(result.set, requires_formula.term, .lt);
    const free = try expectFreeVarTermRef(result.set, requires_body.lhs, 9, 3, "x");
    try expectTypeRefSpelling(free.ty, "u256");

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "x");
    try expectTypeRefSpelling(outer.binder.ty, "u256");
    const condition_id = outer.condition orelse return error.TestUnexpectedResult;
    const condition = try expectBinaryTerm(result.set, condition_id, .lt);
    try expectBoundVarTerm(result.set, condition.lhs, 0, "x");
    const body = try expectBinaryTerm(result.set, outer.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 0, "x");
}

test "formal obligation MLIR adapter derives term types from MLIR values" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @typed_terms(%arg0: !ora.int<64, true>, %arg1: !ora.int<32, false>, %arg2: !ora.address) attributes {
        \\    ora.param_names = ["sx", "uy", "who"],
        \\    ora.param_binding_ids = ["file:11:pattern:1", "file:11:pattern:2", "file:11:pattern:3"]
        \\  } {
        \\    %req = ora.cmp "ule", %arg1, %arg1 : !ora.int<32, false>, !ora.int<32, false> -> i1
        \\    "ora.requires"(%req) : (i1) -> ()
        \\    %addr = ora.cmp "eq", %arg2, %arg2 : !ora.address, !ora.address -> i1
        \\    "ora.assume"(%addr) : (i1) -> ()
        \\    %ens = ora.cmp "le", %arg0, %arg0 : !ora.int<64, true>, !ora.int<64, true> -> i1
        \\    "ora.ensures"(%ens) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .assume));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    for (result.set.assumptions) |assumption| {
        const formula = assumption.formula orelse return error.TestUnexpectedResult;
        try testing.expect(formula == .term);
        const binary = try expectBinaryTerm(result.set, formula.term, if (assumption.kind == .requires) .le else .eq);
        switch (assumption.kind) {
            .requires => {
                const free = try expectFreeVarTermRef(result.set, binary.lhs, 11, 2, "uy");
                try expectTypeRefSpelling(free.ty, "u32");
                const rhs_free = try expectFreeVarTermRef(result.set, binary.rhs, 11, 2, "uy");
                try expectTypeRefSpelling(rhs_free.ty, "u32");
            },
            .assume => {
                const free = try expectFreeVarTermRef(result.set, binary.lhs, 11, 3, "who");
                try expectTypeRefSpelling(free.ty, "address");
            },
            else => return error.TestUnexpectedResult,
        }
    }

    const sx = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "sx");
    try expectTypeRefSpelling(sx.binder.ty, "i64");
    const uy = try expectQuantifiedTerm(result.set, sx.body, .forall, "uy");
    try expectTypeRefSpelling(uy.binder.ty, "u32");
    const who = try expectQuantifiedTerm(result.set, uy.body, .forall, "who");
    try expectTypeRefSpelling(who.binder.ty, "address");
}

test "formal obligation MLIR adapter tracks nested forall binders with De Bruijn indexes" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @nested_forall() {
        \\    %i = arith.constant {ora.bound_variable = "i"} 0 : i256
        \\    %j = arith.constant {ora.bound_variable = "j"} 0 : i256
        \\    %cmp = arith.cmpi ule, %i, %j : i256
        \\    %inner = "ora.quantified"(%cmp) <{quantifier = "forall", variable = "j", variable_type = "u256"}> : (i1) -> i1
        \\    %outer = "ora.quantified"(%inner) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1) -> i1
        \\    "ora.assert"(%outer) <{message = "nested"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .assert), .forall, "i");
    const inner = try expectQuantifiedTerm(result.set, outer.body, .forall, "j");
    const body = try expectBinaryTerm(result.set, inner.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 1, "i");
    try expectBoundVarTerm(result.set, body.rhs, 0, "j");
}

test "formal obligation MLIR adapter resolves shadowed forall binders to nearest binder" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @shadowed_forall() {
        \\    %i = arith.constant {ora.bound_variable = "i"} 0 : i256
        \\    %cmp = arith.cmpi ule, %i, %i : i256
        \\    %inner = "ora.quantified"(%cmp) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1) -> i1
        \\    %outer = "ora.quantified"(%inner) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1) -> i1
        \\    "ora.assert"(%outer) <{message = "shadowed"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .assert), .forall, "i");
    const inner = try expectQuantifiedTerm(result.set, outer.body, .forall, "i");
    const body = try expectBinaryTerm(result.set, inner.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 0, "i");
    try expectBoundVarTerm(result.set, body.rhs, 0, "i");
}

test "formal obligation MLIR adapter records effect frame summaries" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface() attributes {
        \\    ora.effect = "readwrites",
        \\    ora.modifies_slots = ["balances[param#0]", "config.owner"],
        \\    ora.read_slots = ["balances[param#0]", "transient:scratch"],
        \\    ora.write_slots = ["balances[param#0]"]
        \\  } {
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 3), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 2), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_frame));

    const writes = result.set.obligations[0].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.write_covered_by_modifies, writes.relation);
    try testing.expectEqual(@as(usize, 2), writes.declared.len);
    try testing.expectEqual(@as(usize, 1), writes.actual.len);
    try testing.expectEqualStrings("balances", writes.declared[0].root);
    try testing.expectEqual(obligation.RegionRef.storage, writes.declared[0].region);
    try testing.expectEqual(@as(usize, 1), writes.declared[0].keys.len);
    try testing.expect(writes.declared[0].keys[0] == .parameter);
    try testing.expectEqual(@as(u32, 0), writes.declared[0].keys[0].parameter);
    try testing.expectEqualStrings("config", writes.declared[1].root);
    try testing.expectEqual(@as(usize, 1), writes.declared[1].fields.len);
    try testing.expectEqualStrings("owner", writes.declared[1].fields[0]);

    const reads = result.set.obligations[1].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.read_preserved_by_frame, reads.relation);
    try testing.expectEqual(@as(usize, 1), reads.declared.len);
    try testing.expectEqual(@as(usize, 1), reads.actual.len);
    try testing.expectEqualStrings("scratch", reads.actual[0].root);
    try testing.expectEqual(obligation.RegionRef.transient, reads.actual[0].region);
}

test "formal obligation MLIR adapter records lock and external frame metadata" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func private @callee()
        \\  func.func @lock_and_call(%resource: i256) {
        \\    ora.lock %resource {key = "balances[msg.sender]"} : i256
        \\    func.call @callee() {ora.trusted_extern_frame = "caller_storage"} : () -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 3), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .lock_covers_write));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .external_call_frame));

    const lock = result.set.obligations[0].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.lock_covers_write, lock.relation);
    try testing.expectEqual(@as(usize, 1), lock.declared.len);
    try testing.expectEqualStrings("balances", lock.declared[0].root);
    try testing.expectEqual(@as(usize, 1), lock.declared[0].keys.len);
    try testing.expect(lock.declared[0].keys[0] == .msg_sender);

    const external = result.set.obligations[1].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.external_call_frame, external.relation);
    try testing.expectEqual(@as(usize, 1), external.declared.len);
    try testing.expectEqual(obligation.RegionRef.none, external.declared[0].region);
    try testing.expectEqualStrings("caller_storage", external.declared[0].root);
}

test "formal obligation MLIR adapter covers Z3 assertion tags and implicit safety ops" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @z3_surface(%flag: i1, %x: i256, %y: i256) {
        \\    "cf.assert"(%flag) <{msg = "requires"}> : (i1) -> ()
        \\    "ora.assert"(%flag) : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    %q = arith.divui %x, %y : i256
        \\    %s = arith.shli %x, %y : i256
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const cf_assert = findFirstOpByName(module, "cf.assert") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        cf_assert,
        strRef("ora.requires"),
        mlir.oraBoolAttrCreate(h.ctx, true),
    );

    const ora_assert = findFirstOpByName(module, "ora.assert") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        ora_assert,
        strRef("ora.verification_type"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard")),
    );
    mlir.oraOperationSetAttributeByName(
        ora_assert,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:z3_surface:flag")),
    );

    const assume_op = findFirstOpByName(module, "ora.assume") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        assume_op,
        strRef("ora.assume_origin"),
        mlir.oraStringAttrCreate(h.ctx, strRef("path")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 3), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 5), result.set.queries.len);
    try testing.expectEqual(@as(usize, 0), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .path_assume));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .arithmetic_safety));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 2), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));

    for (result.set.obligations) |item| {
        if (item.kind == .logical and item.kind.logical.role == .arithmetic_safety) {
            try testing.expectEqual(obligation.ValueRefKind.derived, item.kind.logical.formula.origin_value.kind);
        }
    }
}

test "formal obligation MLIR adapter expands resource op properties" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @transfer_resource(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operand_segment_sizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 6), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 7), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 6), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .amount_non_negative));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .source_sufficient));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .destination_no_overflow));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .same_place_net_zero));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .conservation));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .modifies_covered));

    for (result.set.obligations) |item| {
        try testing.expect(item.kind == .resource);
        try testing.expectEqualStrings("TokenUnit", item.kind.resource.domain);
        try testing.expect(item.kind.resource.amount.? == .term);
        const source = item.kind.resource.source orelse return error.TestUnexpectedResult;
        const destination = item.kind.resource.destination orelse return error.TestUnexpectedResult;
        try expectPlaceRoot(source, "arg#0", .storage);
        try expectPlaceRoot(destination, "arg#0", .storage);
        try testing.expectEqual(@as(usize, 1), source.keys.len);
        try testing.expectEqual(@as(usize, 1), destination.keys.len);
        try expectParameterKey(source.keys[0], 1);
        try expectParameterKey(destination.keys[0], 2);
    }
}

test "formal obligation MLIR adapter records direct resource create and destroy places" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @R {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @direct(%amount: !ora.int<256, false>) {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      "ora.create"(%reserve, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>) -> ()
        \\      %scratch = ora.tload "scratch" : !ora.int<256, false>
        \\      "ora.destroy"(%scratch, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 6), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .create, .destination_no_overflow));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .destroy, .source_sufficient));

    var saw_create = false;
    var saw_destroy = false;
    for (result.set.obligations) |item| {
        if (item.kind != .resource) continue;
        switch (item.kind.resource.op) {
            .create => if (!saw_create) {
                saw_create = true;
                const destination = item.kind.resource.destination orelse return error.TestUnexpectedResult;
                try expectPlaceRoot(destination, "reserve", .storage);
                try testing.expectEqual(@as(usize, 0), destination.keys.len);
            },
            .destroy => if (!saw_destroy) {
                saw_destroy = true;
                const source = item.kind.resource.source orelse return error.TestUnexpectedResult;
                try expectPlaceRoot(source, "scratch", .transient);
                try testing.expectEqual(@as(usize, 0), source.keys.len);
            },
            .move => {},
        }
    }
    try testing.expect(saw_create);
    try testing.expect(saw_destroy);
}

test "formal obligation MLIR adapter records quantifier metadata" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bounded(%flag: i1) {
        \\    %q = "ora.quantified"(%flag, %flag) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1, i1) -> i1
        \\    "ora.assert"(%q) <{message = "bounded"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));

    var saw_quantifier = false;
    for (result.set.obligations) |item| {
        if (item.kind != .quantifier) continue;
        saw_quantifier = true;
        const quantifier = item.kind.quantifier;
        try testing.expectEqual(obligation.ArtifactPolicy.diagnostic_only, item.artifact_policy);
        try testing.expectEqual(obligation.Quantifier.forall, quantifier.quantifier);
        try testing.expectEqualStrings("i", quantifier.variable);
        try testing.expectEqualStrings("u256", quantifier.binder_type.spelling);
        try testing.expectEqual(obligation.QuantifierBinderSort.bit_vector, quantifier.binder_sort);
        try testing.expectEqual(obligation.VerificationQueryFragment.aufbv_quantifiers, quantifier.fragment);
        try testing.expectEqual(obligation.QuantifierPatternStatus.absent, quantifier.pattern_status);
        try testing.expectEqual(@as(?obligation.QuantifierDegradation, null), quantifier.degradation);
    }
    try testing.expect(saw_quantifier);
}

test "formal report coverage summarizes representative MLIR obligation classes" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @coverage(%flag: i1, %balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>, %x: i256, %y: i256) attributes {
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#0]"],
        \\    ora.write_slots = ["balances[param#0]"]
        \\  } {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked addition overflow"}> : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operand_segment_sizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    %q = "ora.quantified"(%flag, %flag) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1, i1) -> i1
        \\    "ora.assert"(%q) <{message = "bounded"}> : (i1) -> ()
        \\    %d = arith.divui %x, %y : i256
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:coverage:flag")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 13), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 14), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .arithmetic_safety));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .conservation));
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 11), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);

    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"artifact_decision\",\"schema_version\":1,\"status\":\"blocked\",\"reason\":\"missing_proof\"}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":1,\"assumptions\":1,\"obligations\":13,\"queries\":14"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_obligation_links\":13"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"obligation_kinds\":{\"logical\":4,\"runtime_guard\":1,\"type_wf\":0,\"type_relation\":0,\"region_relation\":0,\"effect_frame\":1,\"resource\":6,\"quantifier\":1,\"filtered_input\":0,\"backend_fact\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_backends\":{\"unspecified\":14,\"z3\":0,\"lean\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_results\":{\"missing\":14,\"sat\":0,\"unsat\":0,\"unknown\":0,\"proved\":0,\"failed\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"assumption\",\"schema_version\":1"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"logical\",\"role\":\"ensures\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"addition_overflow\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"division_by_zero\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"runtime_guard\",\"guard_id\":\"guard:coverage:flag\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"effect_frame\",\"relation\":\"write_covered_by_modifies\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"resource\",\"op\":\"move\",\"domain\":\"TokenUnit\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"property\":\"conservation\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"quantifier\",\"quantifier\":\"forall\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"query\",\"schema_version\":1"));
}

test "formal report coverage includes representative source contracts" {
    var logical = try collectPackageObligations(testing.allocator, "ora-example/formal/obligation_report_logical.ora");
    defer logical.deinit();
    try testing.expect(countAssumption(logical.set, .requires) > 0);
    try testing.expect(countLogical(logical.set, .ensures) > 0);
    try testing.expect(countLogical(logical.set, .assert) > 0);
    try testing.expect(countArithmeticSafety(logical.set, .subtraction_overflow) > 0);
    {
        const report = try dumpManifestToOwnedString(testing.allocator, logical.set);
        defer testing.allocator.free(report);
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":1"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"assumption\",\"schema_version\":1"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"role\":\"ensures\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"role\":\"assert\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"subtraction_overflow\""));
    }

    var resource = try collectPackageObligations(testing.allocator, "ora-example/formal/obligation_report_resource.ora");
    defer resource.deinit();
    try testing.expect(countResource(resource.set, .move, .source_sufficient) > 0);
    try testing.expect(countResource(resource.set, .move, .destination_no_overflow) > 0);
    try testing.expect(countResource(resource.set, .move, .conservation) > 0);
    try testing.expect(countEffectFrame(resource.set, .write_covered_by_modifies) > 0);
    {
        const report = try dumpManifestToOwnedString(testing.allocator, resource.set);
        defer testing.allocator.free(report);
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":1"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"effect_frame\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"resource\",\"op\":\"move\",\"domain\":\"TokenUnit\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"property\":\"conservation\""));
    }

    var quantifier = try collectPackageObligations(testing.allocator, "ora-example/formal/obligation_report_quantifier.ora");
    defer quantifier.deinit();
    try testing.expect(countQuantifier(quantifier.set) > 0);
    {
        const report = try dumpManifestToOwnedString(testing.allocator, quantifier.set);
        defer testing.allocator.free(report);
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":1"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"quantifier\",\"quantifier\":\"forall\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"binder_type\":{\"tag\":\"spelling\",\"value\":\"u256\"}"));
    }
}

test "formal obligation MLIR adapter blocks malformed quantifier binder width" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%flag: i1) {
        \\    %q = "ora.quantified"(%flag) <{quantifier = "exists", variable = "i", variable_type = "uabc"}> : (i1) -> i1
        \\    "ora.assert"(%q) <{message = "bad"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
    try testing.expectEqual(obligation.QuantifierDegradation.malformed_binder_width, result.set.obligations[0].kind.quantifier.degradation.?);
    try testing.expectEqual(obligation.ArtifactBlockReason.blocking_diagnostic, result.set.artifactDecision().blocked);
}

test "formal obligation MLIR adapter blocks unsupported quantifier binder type" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%flag: i1) {
        \\    %q = "ora.quantified"(%flag) <{quantifier = "forall", variable = "item", variable_type = "FutureType"}> : (i1) -> i1
        \\    "ora.assert"(%q) <{message = "bad"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
    try testing.expectEqual(obligation.QuantifierBinderSort.opaque_unknown, result.set.obligations[0].kind.quantifier.binder_sort);
    try testing.expectEqual(obligation.QuantifierDegradation.unsupported_binder_type, result.set.obligations[0].kind.quantifier.degradation.?);
}

test "formal resource move without segments is rejected before obligation collection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    try testing.expectError(error.MlirParseFailed, parseModule(h.ctx, text));
}

test "formal obligation MLIR adapter fails closed on malformed guard marker" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%flag: i1) {
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 0), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
}
