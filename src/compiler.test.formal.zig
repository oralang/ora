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
const type_builtin = @import("ora_types").builtin;

const test_helpers = @import("compiler.test.helpers.zig");
const compilePackage = test_helpers.compilePackage;
const renderOraMlirForSource = test_helpers.renderOraMlirForSource;

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

fn findResourceQuery(
    set: obligation.ObligationSet,
    op: obligation.ResourceOperation,
    property: obligation.ResourceProperty,
) !obligation.VerificationQuery {
    var resource_id: ?obligation.Id = null;
    for (set.obligations) |item| {
        if (item.kind == .resource and item.kind.resource.op == op and item.kind.resource.property == property) {
            resource_id = item.id;
            break;
        }
    }
    const id = resource_id orelse return error.TestUnexpectedResult;
    for (set.queries) |query| {
        if (query.obligation_ids.len == 1 and query.obligation_ids[0] == id) return query;
    }
    return error.TestUnexpectedResult;
}

fn expectSingleResourceSemanticSupport(
    terms: []const obligation.Term,
    goal: obligation.ResourceGoal,
    expected_supported: bool,
) !void {
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "resource_test" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .{ .resource_op = .{ .op = goal.op, .domain = goal.domain, .ordinal = 0 } },
        .kind = .{ .resource = goal },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "resource_test" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };
    const supported = switch (obligation_to_lean.querySemanticSupport(set, queries[0])) {
        .supported => true,
        .unsupported => false,
    };
    try testing.expectEqual(expected_supported, supported);
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

fn expectParameterKey(key: obligation.PlaceKey, expected: obligation.FreeVarId) !void {
    try testing.expect(key == .parameter);
    try testing.expect(obligation.freeVarIdEql(expected, key.parameter));
}

fn expectAnyParameterKey(key: obligation.PlaceKey) !obligation.FreeVarId {
    try testing.expect(key == .parameter);
    return key.parameter;
}

fn expectConstantKey(key: obligation.PlaceKey, expected: []const u8) !void {
    try testing.expect(key == .constant);
    try testing.expectEqualStrings(expected, key.constant);
}

fn expectPlaceRoot(place: obligation.PlaceRef, expected_root: []const u8, expected_region: obligation.RegionRef) !void {
    try testing.expectEqualStrings(expected_root, place.root);
    try testing.expectEqual(expected_region, place.region);
}

fn expectPlaceReadTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_region: obligation.RegionRef,
) !void {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .place_read);
    try expectPlaceRoot(set.terms[term_id].place_read, expected_root, expected_region);
}

fn expectPlaceReadTermWithParameterKeys(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_keys: []const obligation.FreeVarId,
) !void {
    try expectPlaceReadTerm(set, term_id, expected_root, .storage);
    const place = set.terms[term_id].place_read;
    try testing.expectEqual(expected_keys.len, place.keys.len);
    for (expected_keys, 0..) |expected, index| {
        try expectParameterKey(place.keys[index], expected);
    }
}

fn expectPlaceReadTermWithParameterKeyCount(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_count: usize,
) !void {
    try expectPlaceReadTerm(set, term_id, expected_root, .storage);
    const place = set.terms[term_id].place_read;
    try testing.expectEqual(expected_count, place.keys.len);
    for (place.keys) |key| {
        _ = try expectAnyParameterKey(key);
    }
}

fn expectPlaceReadTermWithConstantKeys(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_keys: []const []const u8,
) !void {
    try expectPlaceReadTerm(set, term_id, expected_root, .storage);
    const place = set.terms[term_id].place_read;
    try testing.expectEqual(expected_keys.len, place.keys.len);
    for (expected_keys, 0..) |expected, index| {
        try expectConstantKey(place.keys[index], expected);
    }
}

fn expectOldPlaceReadTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_region: obligation.RegionRef,
) !void {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .old);
    try expectPlaceReadTerm(set, set.terms[term_id].old, expected_root, expected_region);
}

fn countOldTerms(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.terms) |term| {
        if (term == .old) count += 1;
    }
    return count;
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

fn expectTypeRefBuiltin(ty: ?obligation.TypeRef, expected: type_builtin.BuiltinTypeId) !void {
    const actual = ty orelse return error.TestUnexpectedResult;
    try testing.expect(actual == .compiler_type_id);
    try testing.expectEqual(type_builtin.lookupBuiltinById(expected).comptime_type_id, actual.compiler_type_id);
}

fn builtinTypeRef(expected: type_builtin.BuiltinTypeId) obligation.TypeRef {
    return .{ .compiler_type_id = type_builtin.lookupBuiltinById(expected).comptime_type_id };
}

fn checkFormalZ3TermObligation(
    terms: []const obligation.Term,
    formula: obligation.TermId,
) !obligation_to_z3.CheckStatus {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "adapter_fixture" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "adapter_fixture", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = formula },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "adapter_fixture" } },
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
        .terms = terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    return adapter.checkObligation(1);
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

fn optionalStringEqualForTest(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs) |left| {
        if (rhs) |right| return std.mem.eql(u8, left, right);
        return false;
    }
    return rhs == null;
}

fn ownerEqualForTest(lhs: obligation.Owner, rhs: obligation.Owner) bool {
    return switch (lhs) {
        .module => |left| switch (rhs) {
            .module => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .function => |left| switch (rhs) {
            .function => |right| std.mem.eql(u8, left.name, right.name) and
                optionalStringEqualForTest(left.module, right.module) and
                optionalStringEqualForTest(left.contract, right.contract),
            else => false,
        },
        .contract => |left| switch (rhs) {
            .contract => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .trait_method => |left| switch (rhs) {
            .trait_method => |right| std.mem.eql(u8, left.trait_name, right.trait_name) and
                std.mem.eql(u8, left.method_name, right.method_name) and
                optionalStringEqualForTest(left.impl_name, right.impl_name),
            else => false,
        },
        .statement => |left| switch (rhs) {
            .statement => |right| left.ordinal == right.ordinal and
                std.mem.eql(u8, left.function_name, right.function_name),
            else => false,
        },
        .backend => |left| switch (rhs) {
            .backend => |right| left.component == right.component and std.mem.eql(u8, left.name, right.name),
            else => false,
        },
    };
}

fn obligationOwnerByIdForTest(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Owner {
    for (set.obligations) |item| {
        if (item.id == id) return item.owner;
    }
    return null;
}

fn assumptionOwnerByIdForTest(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Owner {
    for (set.assumptions) |item| {
        if (item.id == id) return item.owner;
    }
    return null;
}

fn expectQueriesOwnerScoped(set: obligation.ObligationSet) !void {
    for (set.queries) |query| {
        for (query.obligation_ids) |id| {
            const owner = obligationOwnerByIdForTest(set, id) orelse return error.TestUnexpectedResult;
            try testing.expect(ownerEqualForTest(query.owner, owner));
        }
        for (query.assumption_ids) |id| {
            const owner = assumptionOwnerByIdForTest(set, id) orelse return error.TestUnexpectedResult;
            try testing.expect(ownerEqualForTest(query.owner, owner));
        }
    }
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

fn runProcessWithLeanPath(allocator: std.mem.Allocator, args: []const []const u8) !std.process.RunResult {
    const path_arg = try leanPathEnvArgForTest(allocator);
    defer allocator.free(path_arg);

    var argv = try allocator.alloc([]const u8, args.len + 2);
    defer allocator.free(argv);
    argv[0] = "/usr/bin/env";
    argv[1] = path_arg;
    @memcpy(argv[2..], args);
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

fn findLogicalQuery(set: obligation.ObligationSet, role: obligation.LogicalRole) !obligation.VerificationQuery {
    for (set.queries) |query| {
        if (query.kind == .obligation and query.logical_role != null and query.logical_role.? == role) return query;
    }
    return error.TestUnexpectedResult;
}

fn expectLogicalQueryUnsupported(
    set: obligation.ObligationSet,
    role: obligation.LogicalRole,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findLogicalQuery(set, role))) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
}

fn expectLogicalQuerySupported(set: obligation.ObligationSet, role: obligation.LogicalRole) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findLogicalQuery(set, role))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
}

fn expectEnsuresQueryUnsupported(
    set: obligation.ObligationSet,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEnsuresQuery(set))) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
}

fn expectEnsuresQuerySupported(set: obligation.ObligationSet) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEnsuresQuery(set))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
}

fn findEffectFrameQuery(set: obligation.ObligationSet, relation: obligation.EffectFrameRelation) !obligation.VerificationQuery {
    for (set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != relation) continue;
        for (set.queries) |query| {
            for (query.obligation_ids) |id| {
                if (id == item.id) return query;
            }
        }
    }
    return error.TestUnexpectedResult;
}

fn expectEffectFrameQuerySupported(set: obligation.ObligationSet, relation: obligation.EffectFrameRelation) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEffectFrameQuery(set, relation))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
}

fn expectEffectFrameQueryUnsupported(
    set: obligation.ObligationSet,
    relation: obligation.EffectFrameRelation,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEffectFrameQuery(set, relation))) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
}

fn expectSyntheticKeyEvidenceUnsupported(
    terms: []const obligation.Term,
    formula_term: obligation.TermId,
    assumption_kind: obligation.AssumptionKind,
    query_assumption_ids: []const obligation.Id,
    assumption_owner: obligation.Owner,
    query_owner: obligation.Owner,
    read: obligation.PlaceRef,
    write: obligation.PlaceRef,
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    const assumption_id: obligation.Id = 10;
    const obligation_id: obligation.Id = 20;
    const obligation_ids = [_]obligation.Id{obligation_id};
    const assumptions = [_]obligation.Assumption{.{
        .id = assumption_id,
        .owner = assumption_owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = assumption_kind,
        .formula = .{ .term = formula_term },
    }};
    const evidence = [_]obligation.KeyDisjointEvidence{.{
        .kind = .free_var_disequality,
        .assumption_id = assumption_id,
        .lhs = lhs,
        .rhs = rhs,
        .read = read,
        .write = write,
        .key_index = 0,
    }};
    const declared = [_]obligation.PlaceRef{write};
    const actual = [_]obligation.PlaceRef{read};
    const obligations = [_]obligation.Obligation{.{
        .id = obligation_id,
        .owner = query_owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .{ .effect_frame = .{
            .relation = .read_preserved_by_key_evidence,
            .declared = &declared,
            .actual = &actual,
            .evidence = &evidence,
        } },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 30,
        .owner = query_owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .assumption_ids = query_assumption_ids,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &queries,
        .terms = terms,
    };

    switch (obligation_to_lean.querySemanticSupport(set, queries[0])) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
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
    try writeProofManifestIdsForTest(
        allocator,
        path,
        module_name,
        theorem_name,
        proof_path,
        query.id,
        query.obligation_ids,
        query.assumption_ids,
    );
}

fn writeProofManifestIdsForTest(
    allocator: std.mem.Allocator,
    path: []const u8,
    module_name: []const u8,
    theorem_name: []const u8,
    proof_path: []const u8,
    query_id: obligation.Id,
    obligation_ids: []const obligation.Id,
    assumption_ids: []const obligation.Id,
) !void {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\n  \"schema_version\": 1,\n  \"proofs\": [\n    {\n      \"query_id\": ");
    try writer.print("{d}", .{query_id});
    try writer.writeAll(",\n      \"obligation_ids\": ");
    try writeIdArrayForTest(writer, obligation_ids);
    try writer.writeAll(",\n      \"assumption_ids\": ");
    try writeIdArrayForTest(writer, assumption_ids);
    try writer.writeAll(",\n      \"module_name\": ");
    try writeJsonStringForTest(writer, module_name);
    try writer.writeAll(",\n      \"theorem_name\": ");
    try writeJsonStringForTest(writer, theorem_name);
    try writer.writeAll(",\n      \"path\": ");
    try writeJsonStringForTest(writer, proof_path);
    try writer.writeAll(",\n      \"content_sha256\": null\n    }\n  ]\n}\n");

    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = out.written() });
}

fn writeTwoProofManifestForTest(
    allocator: std.mem.Allocator,
    path: []const u8,
    module_name: []const u8,
    first_theorem: []const u8,
    first_proof_path: []const u8,
    first_query: obligation.VerificationQuery,
    second_theorem: []const u8,
    second_proof_path: []const u8,
    second_query: obligation.VerificationQuery,
) !void {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\n  \"schema_version\": 1,\n  \"proofs\": [\n");
    inline for (.{ .{ first_theorem, first_proof_path, first_query }, .{ second_theorem, second_proof_path, second_query } }, 0..) |row, index| {
        if (index != 0) try writer.writeAll(",\n");
        try writer.writeAll("    {\n      \"query_id\": ");
        try writer.print("{d}", .{row[2].id});
        try writer.writeAll(",\n      \"obligation_ids\": ");
        try writeIdArrayForTest(writer, row[2].obligation_ids);
        try writer.writeAll(",\n      \"assumption_ids\": ");
        try writeIdArrayForTest(writer, row[2].assumption_ids);
        try writer.writeAll(",\n      \"module_name\": ");
        try writeJsonStringForTest(writer, module_name);
        try writer.writeAll(",\n      \"theorem_name\": ");
        try writeJsonStringForTest(writer, row[0]);
        try writer.writeAll(",\n      \"path\": ");
        try writeJsonStringForTest(writer, row[1]);
        try writer.writeAll(",\n      \"content_sha256\": null\n    }");
    }
    try writer.writeAll("\n  ]\n}\n");

    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = out.written() });
}

const ProofRecipeRowForTest = struct {
    query_id: obligation.Id,
    obligation_ids: []obligation.Id,
    assumption_ids: []obligation.Id,

    fn deinit(self: *ProofRecipeRowForTest, allocator: std.mem.Allocator) void {
        allocator.free(self.obligation_ids);
        allocator.free(self.assumption_ids);
    }
};

fn parseIdListForTest(allocator: std.mem.Allocator, text: []const u8) ![]obligation.Id {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return try allocator.alloc(obligation.Id, 0);

    var ids = std.ArrayList(obligation.Id).empty;
    errdefer ids.deinit(allocator);

    var it = std.mem.splitScalar(u8, trimmed, ',');
    while (it.next()) |part| {
        const item = std.mem.trim(u8, part, " \t\r\n");
        if (item.len == 0) return error.TestUnexpectedResult;
        try ids.append(allocator, try std.fmt.parseInt(obligation.Id, item, 10));
    }
    return try ids.toOwnedSlice(allocator);
}

fn parseBracketedIdListForTest(
    allocator: std.mem.Allocator,
    text: []const u8,
    marker: []const u8,
    start: usize,
) ![]obligation.Id {
    const marker_start = std.mem.indexOfPos(u8, text, start, marker) orelse return error.TestUnexpectedResult;
    const list_start = marker_start + marker.len;
    const list_end = std.mem.indexOfScalarPos(u8, text, list_start, ']') orelse return error.TestUnexpectedResult;
    return try parseIdListForTest(allocator, text[list_start..list_end]);
}

fn parseProofRecipeRowForTest(allocator: std.mem.Allocator, text: []const u8) !ProofRecipeRowForTest {
    const marker = "proof row: query_id=";
    const row_start = std.mem.indexOf(u8, text, marker) orelse return error.TestUnexpectedResult;
    const query_start = row_start + marker.len;
    const query_end = std.mem.indexOfScalarPos(u8, text, query_start, ',') orelse return error.TestUnexpectedResult;
    const query_id = try std.fmt.parseInt(obligation.Id, text[query_start..query_end], 10);

    const obligation_ids = try parseBracketedIdListForTest(allocator, text, "obligation_ids=[", query_end);
    errdefer allocator.free(obligation_ids);
    const assumption_ids = try parseBracketedIdListForTest(allocator, text, "assumption_ids=[", query_end);
    errdefer allocator.free(assumption_ids);

    return .{
        .query_id = query_id,
        .obligation_ids = obligation_ids,
        .assumption_ids = assumption_ids,
    };
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
            \\      TyRef.isU256,
            \\      TyRef.isU256Carrier,
            \\      compilerTypeIdU256,
            \\      compilerTypeIdI256,
            \\      compilerTypeIdBool,
            \\      Ora.Spec.expectedCompilerTypeIdU256,
            \\      Ora.Spec.expectedCompilerTypeIdI256,
            \\      Ora.Spec.expectedCompilerTypeIdBool
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

fn storageProofModuleFromGeneratedObligations(
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
            \\  · refine ⟨Env.empty, ?_⟩
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      emittedManifest,
            \\      emittedAssumptions
            \\    ]
            \\  · intro env hAssumptions
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      Env.lookupPlace
            \\    ]
            \\    exact stable_place_read_self_eq_denotes env
            \\      { root := "balance", region := .storage, fields := [], keys := [] }
            \\
            \\
        );
    }
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn storagePlaceProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    place_expr: []const u8,
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
    try writer.writeAll("  unfold ");
    try writer.print("emittedQuery_{d}", .{query_id});
    try writer.writeAll(
        \\ obligationFollowsFromAssumptions
        \\  constructor
        \\  · refine ⟨Env.empty, ?_⟩
        \\    simp [
        \\      assumptionsDenoteInEnv,
        \\      assumptionsDenoteInEnv?,
        \\      assumptionAnd?,
        \\      assumptionDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      emittedManifest,
        \\      emittedAssumptions
        \\    ]
        \\  · intro env hAssumptions
        \\    simp [
        \\      obligationDenotesInEnv,
        \\      obligationDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      denoteValue?,
        \\      emittedManifest,
        \\      emittedTerms,
        \\      Env.lookupPlace
        \\    ]
        \\    exact stable_place_read_self_eq_denotes env
    );
    try writer.writeByte(' ');
    try writer.writeAll(place_expr);
    try writer.writeAll(
        \\
        \\
    );

    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn writeLeanFreeVarIdForTest(writer: anytype, id: obligation.FreeVarId) !void {
    try writer.print("{{ file_id := {d}, pattern_id := {d} }}", .{ id.file_id, id.pattern_id });
}

fn writeLeanFreeVarEqFactForTest(
    writer: anytype,
    indent: []const u8,
    name: []const u8,
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
) !void {
    try writer.writeAll(indent);
    try writer.writeAll("have ");
    try writer.writeAll(name);
    try writer.writeAll(" : ((");
    try writeLeanFreeVarIdForTest(writer, lhs);
    try writer.writeAll(" : FreeVarId) == ");
    try writeLeanFreeVarIdForTest(writer, rhs);
    try writer.writeAll(") = ");
    try writer.writeAll(if (obligation.freeVarIdEql(lhs, rhs)) "true" else "false");
    try writer.writeAll(" := by\n");
    try writer.writeAll(indent);
    try writer.writeAll("  decide\n");
}

fn writeLeanStringLiteralForTest(writer: anytype, value: []const u8) !void {
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

fn writeLeanRegionRefForTest(writer: anytype, region: obligation.RegionRef) !void {
    try writer.writeAll(switch (region) {
        .none => ".none",
        .storage => ".storage",
        .memory => ".memory",
        .transient => ".transient",
        .calldata => ".calldata",
    });
}

fn writeLeanPlaceKeyForTest(writer: anytype, key: obligation.PlaceKey) !void {
    switch (key) {
        .parameter => |id| {
            try writer.writeAll(".parameter ");
            try writeLeanFreeVarIdForTest(writer, id);
        },
        .comptime_parameter => |index| try writer.print(".comptimeParameter {d}", .{index}),
        .comptime_range_parameter => |index| try writer.print(".comptimeRangeParameter {d}", .{index}),
        .constant => |value| {
            try writer.writeAll(".constant ");
            try writeLeanStringLiteralForTest(writer, value);
        },
        .msg_sender => try writer.writeAll(".msgSender"),
        .tx_origin => try writer.writeAll(".txOrigin"),
        .unknown => try writer.writeAll(".unknown"),
    }
}

fn writeLeanPlaceRefForTest(writer: anytype, place: obligation.PlaceRef) !void {
    try writer.writeAll("{ root := ");
    try writeLeanStringLiteralForTest(writer, place.root);
    try writer.writeAll(", region := ");
    try writeLeanRegionRefForTest(writer, place.region);
    try writer.writeAll(", fields := [");
    for (place.fields, 0..) |field, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLeanStringLiteralForTest(writer, field);
    }
    try writer.writeAll("], keys := [");
    for (place.keys, 0..) |key, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLeanPlaceKeyForTest(writer, key);
    }
    try writer.writeAll("] }");
}

fn writeLeanPlaceEqFactForTest(
    writer: anytype,
    indent: []const u8,
    name: []const u8,
    place: obligation.PlaceRef,
) !void {
    try writer.writeAll(indent);
    try writer.writeAll("have ");
    try writer.writeAll(name);
    try writer.writeAll(" : ((");
    try writeLeanPlaceRefForTest(writer, place);
    try writer.writeAll(" : PlaceRef) == ");
    try writeLeanPlaceRefForTest(writer, place);
    try writer.writeAll(") = true := by\n");
    try writer.writeAll(indent);
    try writer.writeAll("  decide\n");
}

fn keyEvidenceFrameProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    read: obligation.PlaceRef,
    write: obligation.PlaceRef,
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
    write_query_id: ?obligation.Id,
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
        try writer.writeAll(" obligationFollowsFromAssumptions\n");
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hLhsLhs", lhs, lhs);
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hLhsRhs", lhs, rhs);
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hRhsLhs", rhs, lhs);
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hRhsRhs", rhs, rhs);
        try writer.writeAll("  have hRhsNeLhs : ((");
        try writeLeanFreeVarIdForTest(writer, rhs);
        try writer.writeAll(" : FreeVarId) != ");
        try writeLeanFreeVarIdForTest(writer, lhs);
        try writer.writeAll(") = true := by\n    decide\n");
        try writer.writeAll("  have hStorageSelfNe : (RegionRef.storage != RegionRef.storage) = false := by\n    decide\n");
        try writer.writeAll("  have hRequiresSelfNe : (AssumptionKind.requires != AssumptionKind.requires) = false := by\n    decide\n");
        try writer.writeAll("  have hBinaryNeSelfNe : (BinaryOp.ne != BinaryOp.ne) = false := by\n    decide\n");
        try writeLeanPlaceEqFactForTest(writer, "  ", "hReadRead", read);
        try writeLeanPlaceEqFactForTest(writer, "  ", "hWriteWrite", write);
        try writer.writeAll(
            \\  constructor
            \\  · refine ⟨(Env.empty.setFree
        );
        try writer.writeAll(" ");
        try writeLeanFreeVarIdForTest(writer, lhs);
        try writer.writeAll(" (.u256 (BitVec.ofNat 256 0))).setFree ");
        try writeLeanFreeVarIdForTest(writer, rhs);
        try writer.writeAll(
            \\ (.u256 (BitVec.ofNat 256 1)), ?_⟩
            \\    have hNe : ¬(BitVec.ofNat 256 0 : U256) = BitVec.ofNat 256 1 := by
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
            \\      emittedAssumptions,
            \\      Env.setFree,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      hLhsLhs,
            \\      hLhsRhs,
            \\      hRhsLhs,
            \\      hRhsRhs,
            \\      hRhsNeLhs,
            \\      hStorageSelfNe,
            \\      hRequiresSelfNe,
            \\      hBinaryNeSelfNe,
            \\      hReadRead,
            \\      hWriteWrite,
            \\      hNe
            \\    ]
            \\  · intro env hAssumptions
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
            \\      emittedAssumptions,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      hLhsLhs,
            \\      hLhsRhs,
            \\      hRhsLhs,
            \\      hRhsRhs,
            \\      hRhsNeLhs,
            \\      hStorageSelfNe,
            \\      hRequiresSelfNe,
            \\      hBinaryNeSelfNe,
            \\      hReadRead,
            \\      hWriteWrite
            \\    ] at hAssumptions
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      effectFrameGoalDenotes?,
            \\      placeListDisjointWithEvidence?,
            \\      placePairDisjointWithEvidence?,
            \\      placeDefinitelyDisjoint,
            \\      placeKeyListsDefinitelyDisjoint,
            \\      placeKeysDefinitelyDistinct,
            \\      RegionRef.isConcrete,
            \\      computedStorageRoot,
            \\      pairCoveredByEvidence,
            \\      evidenceMatchesPair,
            \\      evidenceListDenotes?,
            \\      keyDisjointEvidenceDenotes?,
            \\      keyEvidencePathMatches,
            \\      keyDisjointEvidenceFormulaDenotes?,
            \\      Manifest.assumptionById,
            \\      termFreeVarId?,
            \\      freeVarPairMatches,
            \\      placeKeysEqualBefore,
            \\      optionPropAnd?,
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      emittedAssumptions,
            \\      emittedObligations,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      TyRef.isU256,
            \\      TyRef.isI256,
            \\      TyRef.isU256Carrier,
            \\      compilerTypeIdU256,
            \\      compilerTypeIdI256,
            \\      Ora.Spec.expectedCompilerTypeIdU256,
            \\      Ora.Spec.expectedCompilerTypeIdI256,
            \\      hLhsLhs,
            \\      hLhsRhs,
            \\      hRhsLhs,
            \\      hRhsRhs,
            \\      hRhsNeLhs,
            \\      hStorageSelfNe,
            \\      hRequiresSelfNe,
            \\      hBinaryNeSelfNe,
            \\      hReadRead,
            \\      hWriteWrite,
            \\      hAssumptions
            \\    ]
            \\    exact option_prop_and_true_left_intro _ hAssumptions
            \\
            \\
        );
    }

    if (write_query_id) |id| {
        try writer.print("theorem discharge_write : emittedQuery_{d} := by\n", .{id});
        try writer.writeAll("  unfold ");
        try writer.print("emittedQuery_{d}", .{id});
        try writer.writeAll(" obligationFollowsFromAssumptions\n");
        try writeLeanPlaceEqFactForTest(writer, "  ", "hWriteWrite", write);
        try writer.writeAll(
            \\  constructor
            \\  · refine ⟨Env.empty, ?_⟩
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?
            \\    ]
            \\  · intro env hAssumptions
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      effectFrameGoalDenotes?,
            \\      placeListCovers,
            \\      emittedManifest,
            \\      emittedObligations,
            \\      hWriteWrite
            \\    ]
            \\
            \\
        );
    }
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn signedComparisonProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
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
    try writer.writeAll("  unfold ");
    try writer.print("emittedQuery_{d}", .{query_id});
    try writer.writeAll(
        \\ obligationFollowsFromAssumptions
        \\  constructor
        \\  · refine ⟨Env.empty, ?_⟩
        \\    simp [
        \\      assumptionsDenoteInEnv,
        \\      assumptionsDenoteInEnv?,
        \\      assumptionAnd?,
        \\      assumptionDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      emittedManifest,
        \\      emittedAssumptions
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
        \\      TyRef.isU256,
        \\      TyRef.isI256,
        \\      TyRef.isU256Carrier,
        \\      compilerTypeIdU256,
        \\      compilerTypeIdI256,
        \\      compilerTypeIdBool,
        \\      Ora.Spec.expectedCompilerTypeIdU256,
        \\      Ora.Spec.expectedCompilerTypeIdI256,
        \\      Ora.Spec.expectedCompilerTypeIdBool,
        \\      U256.sle
        \\    ]
        \\
        \\
    );
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn divRemProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    requires_nonzero_y: bool,
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
    try writer.writeAll("  unfold ");
    try writer.print("emittedQuery_{d}", .{query_id});
    try writer.writeAll(
        \\ obligationFollowsFromAssumptions
        \\  constructor
        \\
    );
    if (requires_nonzero_y) {
        try writer.writeAll(
            \\  · refine ⟨Env.empty.setFree { file_id := 0, pattern_id := 1 } (.u256 (BitVec.ofNat 256 1)), ?_⟩
            \\    have hY :
            \\        (({ file_id := 0, pattern_id := 1 } : FreeVarId) ==
            \\          { file_id := 0, pattern_id := 1 }) = true := by
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
            \\      emittedAssumptions,
            \\      Env.setFree,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      IntegerLiteralTerm.asU256?,
            \\      compilerTypeIdU256,
            \\      Ora.Spec.expectedCompilerTypeIdU256,
            \\      TyRef.isU256,
            \\      TyRef.isU256Carrier,
            \\      hY
            \\    ]
            \\
        );
    } else {
        try writer.writeAll(
            \\  · refine ⟨Env.empty, ?_⟩
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?
            \\    ]
            \\
        );
    }
    try writer.writeAll(
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
        \\      Value.binaryU256?,
        \\      IntegerLiteralTerm.asU256?,
        \\      BinderRef.isU256,
        \\      TyRef.isU256,
        \\      TyRef.isI256,
        \\      TyRef.isU256Carrier,
        \\      compilerTypeIdU256,
        \\      compilerTypeIdI256,
        \\      compilerTypeIdBool,
        \\      Ora.Spec.expectedCompilerTypeIdU256,
        \\      Ora.Spec.expectedCompilerTypeIdI256,
        \\      Ora.Spec.expectedCompilerTypeIdBool,
        \\      U256.udivTotal
        \\    ]
        \\
        \\
    );
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
        .proof_surface = true,
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

test "B6 storage Lean proof fixture proves read-only old collapse without erasing guard" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6StorageLeanProofGate {
        \\    storage var balance: u256;
        \\
        \\    pub fn same_balance() -> bool
        \\        ensures balance == old(balance)
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
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_storage_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_storage_lean_gate.ora");
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
    try expectEnsuresQuerySupported(formal_result.set);

    const forced_query = "obligation:4";
    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 2, ".placeRead { root := \"balance\", region := .storage"));
    try testing.expect(std.mem.indexOf(u8, obligations_source, ".old ") == null);

    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{ensures_query.id});
    defer allocator.free(expected_query_def);

    {
        const result = try runOraWithForcedUnknown(allocator, forced_query, &.{
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
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for Z3 UNKNOWN obligations") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof recipe for Z3 UNKNOWN obligations"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "query: emittedQuery_") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "query: emittedQuery_"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "theorem discharge_q") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "theorem discharge_q"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, generated_namespace) or
            std.mem.containsAtLeast(u8, result.stderr, 1, generated_namespace));
    }
    const fail_report_path = try pathFromTmpAlloc(allocator, tmp, "fail/b6_storage_lean_gate.smt.report.json");
    defer allocator.free(fail_report_path);
    const fail_report = try readFileAllocForTest(allocator, fail_report_path);
    defer allocator.free(fail_report);
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"status\": \"UNKNOWN\""));
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"vacuity_unknown\": false"));
    const fail_lean_obligations_path = try pathFromTmpAlloc(allocator, tmp, "fail/b6_storage_lean_gate.lean.obligations.lean");
    defer allocator.free(fail_lean_obligations_path);
    const fail_lean_obligations = try readFileAllocForTest(allocator, fail_lean_obligations_path);
    defer allocator.free(fail_lean_obligations);
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 2, ".placeRead { root := \"balance\", region := .storage"));
    {
        const result = try runLeanFileForTest(allocator, fail_lean_obligations_path);
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_storage_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6StorageFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6StorageFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6StorageFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const sorry_module = try std.fmt.allocPrint(allocator, "Ora.B6StorageFixture.{s}.Sorry", .{module_suffix});
    defer allocator.free(sorry_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const sorry_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{sorry_module});
    defer allocator.free(sorry_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const sorry_proof_path = try std.fmt.allocPrint(allocator, "{s}/Sorry.lean", .{fixture_dir});
    defer allocator.free(sorry_proof_path);

    const valid_proof = try storageProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    const sorry_proof = try storageProofModuleFromGeneratedObligations(allocator, obligations_source, sorry_module, ensures_query.id, true);
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
        const result = try runOraWithForcedUnknown(allocator, forced_query, &.{
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

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_lean_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_lean_gate.hex");
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
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_lean_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_lean_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);

    {
        const result = try runOraWithForcedUnknown(allocator, forced_query, &.{
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
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b6_storage_lean_gate.hex", .{}));
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b6_storage_lean_gate.lean.proof.json", .{}));
}

test "B6 storage path Lean proof unblocks constant map read old collapse" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6StoragePathLeanProofGate {
        \\    storage buckets: map<u256, u256>;
        \\
        \\    pub fn framed() -> bool
        \\        ensures buckets[2] == old(buckets[2])
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
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_storage_path_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_storage_path_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    _ = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);
    const force_target = "obligation:2";

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 2, "root := \"buckets\""));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 2, "keys := [.constant \"2\"]"));
    try testing.expect(std.mem.indexOf(u8, obligations_source, ".old ") == null);

    var proof_row: ProofRecipeRowForTest = undefined;
    var proof_row_set = false;
    defer if (proof_row_set) proof_row.deinit(allocator);

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
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
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for Z3 UNKNOWN obligations") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof recipe for Z3 UNKNOWN obligations"));
        proof_row = parseProofRecipeRowForTest(allocator, result.stdout) catch try parseProofRecipeRowForTest(allocator, result.stderr);
        proof_row_set = true;
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_storage_path_lean_gate.hex", .{}));

    const fail_lean_obligations_path = try pathFromTmpAlloc(allocator, tmp, "fail/b6_storage_path_lean_gate.lean.obligations.lean");
    defer allocator.free(fail_lean_obligations_path);
    const fail_lean_obligations = try readFileAllocForTest(allocator, fail_lean_obligations_path);
    defer allocator.free(fail_lean_obligations);
    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{proof_row.query_id});
    defer allocator.free(expected_query_def);
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 2, "root := \"buckets\""));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 2, "keys := [.constant \"2\"]"));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6StoragePathFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6StoragePathFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6StoragePathFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const valid_proof = try storagePlaceProofModuleFromGeneratedObligations(
        allocator,
        fail_lean_obligations,
        valid_module,
        proof_row.query_id,
        "{ root := \"buckets\", region := .storage, fields := [], keys := [.constant \"2\"] }",
    );
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestIdsForTest(
        allocator,
        valid_manifest,
        valid_module,
        valid_theorem,
        valid_proof_path,
        proof_row.query_id,
        proof_row.obligation_ids,
        proof_row.assumption_ids,
    );

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
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

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_path_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_path_lean_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_path_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 1"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
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
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_path_lean_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_path_lean_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);
}

test "B6 storage key evidence Lean proof unblocks structural frame gate" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6KeyEvidenceGate {
        \\    storage var balances: map<u256, u256>;
        \\
        \\    pub fn copy_other(user: u256, other: u256)
        \\        modifies balances[user]
        \\        requires user != other
        \\    {
        \\        let observed: u256 = balances[other];
        \\        balances[user] = observed;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_key_evidence_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_key_evidence_gate.ora");
    defer allocator.free(source_path);
    const empty_out = try pathFromTmpAlloc(allocator, tmp, "empty");
    defer allocator.free(empty_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const sorry_out = try pathFromTmpAlloc(allocator, tmp, "sorry");
    defer allocator.free(sorry_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    try testing.expectEqual(@as(usize, 1), countAssumption(formal_result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(formal_result.set, .read_preserved_by_key_evidence));
    const frame_query = try findEffectFrameQuery(formal_result.set, .read_preserved_by_key_evidence);
    try testing.expectEqual(obligation.VerificationBackend.unspecified, frame_query.backend);
    try testing.expect(frame_query.result == null);
    try testing.expectEqual(@as(usize, 1), frame_query.assumption_ids.len);
    try expectEffectFrameQuerySupported(formal_result.set, .read_preserved_by_key_evidence);
    const write_query = try findEffectFrameQuery(formal_result.set, .write_covered_by_modifies);
    try testing.expectEqual(@as(usize, 0), write_query.assumption_ids.len);

    var frame_evidence: ?obligation.KeyDisjointEvidence = null;
    for (formal_result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        try testing.expectEqual(@as(usize, 1), item.kind.effect_frame.evidence.len);
        frame_evidence = item.kind.effect_frame.evidence[0];
    }
    const evidence = frame_evidence orelse return error.TestUnexpectedResult;

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{frame_query.id});
    defer allocator.free(expected_query_def);
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".readPreservedByKeyEvidence"));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, "keyIndex := 0"));

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            reference_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_key_evidence_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_key_evidence_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expect(reference_hex.len > 0);

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6KeyEvidenceFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6KeyEvidenceFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6KeyEvidenceFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const sorry_module = try std.fmt.allocPrint(allocator, "Ora.B6KeyEvidenceFixture.{s}.Sorry", .{module_suffix});
    defer allocator.free(sorry_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const sorry_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{sorry_module});
    defer allocator.free(sorry_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const sorry_proof_path = try std.fmt.allocPrint(allocator, "{s}/Sorry.lean", .{fixture_dir});
    defer allocator.free(sorry_proof_path);

    const valid_proof = try keyEvidenceFrameProofModuleFromGeneratedObligations(
        allocator,
        obligations_source,
        valid_module,
        frame_query.id,
        evidence.read,
        evidence.write,
        evidence.lhs,
        evidence.rhs,
        write_query.id,
        false,
    );
    defer allocator.free(valid_proof);
    const sorry_proof = try keyEvidenceFrameProofModuleFromGeneratedObligations(
        allocator,
        obligations_source,
        sorry_module,
        frame_query.id,
        evidence.read,
        evidence.write,
        evidence.lhs,
        evidence.rhs,
        write_query.id,
        true,
    );
    defer allocator.free(sorry_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = sorry_proof_path, .data = sorry_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    const sorry_manifest = try pathFromTmpAlloc(allocator, tmp, "sorry-proofs.json");
    defer allocator.free(sorry_manifest);
    const empty_manifest = try pathFromTmpAlloc(allocator, tmp, "empty-proofs.json");
    defer allocator.free(empty_manifest);
    const valid_write_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge_write", .{valid_module});
    defer allocator.free(valid_write_theorem);
    const sorry_write_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge_write", .{sorry_module});
    defer allocator.free(sorry_write_theorem);
    try writeTwoProofManifestForTest(
        allocator,
        valid_manifest,
        valid_module,
        valid_theorem,
        valid_proof_path,
        frame_query,
        valid_write_theorem,
        valid_proof_path,
        write_query,
    );
    try writeTwoProofManifestForTest(
        allocator,
        sorry_manifest,
        sorry_module,
        sorry_theorem,
        sorry_proof_path,
        frame_query,
        sorry_write_theorem,
        sorry_proof_path,
        write_query,
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = empty_manifest,
        .data = "{\n  \"schema_version\": 1,\n  \"proofs\": []\n}\n",
    });

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            empty_out,
            "--lean-proofs",
            empty_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "missing_proof") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "missing_proof"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "empty/b6_key_evidence_gate.hex", .{}));

    {
        const result = try runProcessWithLeanPath(allocator, &.{
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

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_key_evidence_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_key_evidence_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_key_evidence_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"proof_count\": 2"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);

    {
        const result = try runProcess(allocator, &.{
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
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b6_key_evidence_gate.hex", .{}));
}

test "B6 storage key evidence proof rejects contradictory assumptions" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6KeyEvidenceContradiction {
        \\    storage var balances: map<u256, u256>;
        \\
        \\    pub fn copy_other(user: u256, other: u256)
        \\        modifies balances[user]
        \\        requires user != other
        \\        requires user == other
        \\    {
        \\        let observed: u256 = balances[other];
        \\        balances[user] = observed;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_key_evidence_contradiction.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_key_evidence_contradiction.ora");
    defer allocator.free(source_path);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    try testing.expectEqual(@as(usize, 2), countAssumption(formal_result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(formal_result.set, .read_preserved_by_key_evidence));
    const frame_query = try findEffectFrameQuery(formal_result.set, .read_preserved_by_key_evidence);
    try testing.expectEqual(@as(usize, 2), frame_query.assumption_ids.len);
    try expectEffectFrameQuerySupported(formal_result.set, .read_preserved_by_key_evidence);

    var frame_evidence: ?obligation.KeyDisjointEvidence = null;
    for (formal_result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        frame_evidence = item.kind.effect_frame.evidence[0];
    }
    const evidence = frame_evidence orelse return error.TestUnexpectedResult;

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6KeyEvidenceContradictionFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6KeyEvidenceContradictionFixture") catch {};

    const proof_module = try std.fmt.allocPrint(allocator, "Ora.B6KeyEvidenceContradictionFixture.{s}.Invalid", .{module_suffix});
    defer allocator.free(proof_module);
    const proof_path = try std.fmt.allocPrint(allocator, "{s}/Invalid.lean", .{fixture_dir});
    defer allocator.free(proof_path);
    const proof = try keyEvidenceFrameProofModuleFromGeneratedObligations(
        allocator,
        obligations_source,
        proof_module,
        frame_query.id,
        evidence.read,
        evidence.write,
        evidence.lhs,
        evidence.rhs,
        null,
        false,
    );
    defer allocator.free(proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = proof_path, .data = proof });

    const result = try runLeanFileForTest(allocator, proof_path);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 1);
}

test "B6 signed comparison Lean proof unblocks source-level unknown" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6SignedLeanProofGate {
        \\    pub fn signed_reflexive(x: i256) -> bool
        \\        ensures x <= x
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_signed_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_signed_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);
    const force_target = try std.fmt.allocPrint(allocator, "obligation:{d}", .{ensures_query.id});
    defer allocator.free(force_target);

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".sle"));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".compilerTypeId 12"));

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for Z3 UNKNOWN obligations"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_signed_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6SignedFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6SignedFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6SignedFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);

    const valid_proof = try signedComparisonProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id);
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
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

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_signed_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_signed_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 1"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);
}

test "B6 div rem Lean proof unblocks source-level unknown" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6DivRemLeanProofGate {
        \\    pub fn div_reflexive(x: u256) -> bool
        \\        ensures (x / 1) == (x / 1)
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_div_rem_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_div_rem_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".div"));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".compilerTypeId 6"));

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for Z3 UNKNOWN obligations"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_div_rem_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6DivRemFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6DivRemFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6DivRemFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);

    const valid_proof = try divRemProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
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

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_div_rem_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_div_rem_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 1"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);
}

test "B6 div rem Lean proof does not discharge missing divisor guard" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6DivRemGuardBoundary {
        \\    pub fn div_reflexive(x: u256, y: u256) -> bool
        \\        ensures (x / y) == (x / y)
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_div_rem_guard_boundary.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_div_rem_guard_boundary.ora");
    defer allocator.free(source_path);
    const blocked_out = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(blocked_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".div"));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6DivRemBoundaryFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6DivRemBoundaryFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6DivRemBoundaryFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);

    const valid_proof = try divRemProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            blocked_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "failed_query") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "failed_query") or
            std.mem.containsAtLeast(u8, result.stdout, 1, "failed to prove contract invariant") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "failed to prove contract invariant"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "blocked/b6_div_rem_guard_boundary.hex", .{}));
}

test "B4 unknown diagnostic rejects Lean recipe for unsupported semantic type" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B4UnsupportedLeanType {
        \\    pub fn small(x: u32) -> bool
        \\        ensures x <= x
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b4_unsupported_lean_type.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b4_unsupported_lean_type.ora");
    defer allocator.free(source_path);
    const out_dir = try pathFromTmpAlloc(allocator, tmp, "b4-out");
    defer allocator.free(out_dir);

    const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report",
        "--out-dir",
        out_dir,
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe unavailable for some Z3 UNKNOWN obligations"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "unsupported Lean semantic type `u32`"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for Z3 UNKNOWN obligations"));
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "b4-out/b4_unsupported_lean_type.lean.obligations.lean", .{}));
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
    try testing.expect(!overlay.set.hasBlockingDiagnostic());
    for (overlay.set.queries) |query| {
        try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
        try testing.expect(query.smtlib_hash != null);
        try testing.expect(query.result != null);
    }
}

test "formal Z3 overlay ignores only clean unmatched proved rows" {
    const clean_rows = [_]z3_verification.PreparedQueryManifestRow{
        .{
            .kind = .Obligation,
            .function_name = "checked",
            .file = "test.ora",
            .line = 1,
            .column = 1,
            .result_status = .unsat,
        },
    };
    var clean_overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, .{}, &clean_rows);
    defer clean_overlay.deinit();
    try testing.expectEqual(@as(usize, 0), clean_overlay.set.diagnostics.len);
    try testing.expect(!clean_overlay.set.hasBlockingDiagnostic());

    const vacuous_rows = [_]z3_verification.PreparedQueryManifestRow{
        .{
            .kind = .Obligation,
            .function_name = "checked",
            .file = "test.ora",
            .line = 1,
            .column = 1,
            .result_status = .unsat,
            .vacuous = true,
        },
    };
    var vacuous_overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, .{}, &vacuous_rows);
    defer vacuous_overlay.deinit();
    try testing.expectEqual(@as(usize, 1), vacuous_overlay.set.diagnostics.len);
    try testing.expect(vacuous_overlay.set.hasBlockingDiagnostic());
}

test "formal Z3 overlay does not require prepared rows for structural effect frames" {
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "effect_surface" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .{ .effect_frame = .{
            .relation = .read_preserved_by_key_evidence,
        } },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "effect_surface" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
    };

    var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, set, &.{});
    defer overlay.deinit();

    try testing.expectEqual(@as(usize, 0), overlay.set.diagnostics.len);
    try testing.expectEqual(@as(usize, 1), overlay.set.queries.len);
    try testing.expectEqual(obligation.VerificationBackend.unspecified, overlay.set.queries[0].backend);
    try testing.expect(overlay.set.queries[0].result == null);
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

test "formal Z3 adapter distinguishes signed and unsigned comparison tags" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const max_u256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = max_u256, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .slt, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .lt, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "signed" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "signed_cmp", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 2 },
            } },
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "unsigned" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "unsigned_cmp", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 3 },
            } },
        },
    };
    const signed_ids = [_]obligation.Id{1};
    const unsigned_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "signed" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &signed_ids,
        },
        .{
            .id = 4,
            .owner = .{ .function = .{ .name = "unsigned" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &unsigned_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try adapter.checkObligation(1));
    try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, try adapter.checkObligation(2));
}

test "formal Z3 adapter totalizes unsigned div and rem by zero" {
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = "7", .ty = builtinTypeRef(.u256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.u256) } },
        .{ .binary = .{ .op = .div, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
        .{ .binary = .{ .op = .mod, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.u256) } },
        .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 4 } },
        .{ .binary = .{ .op = .eq, .lhs = 3, .rhs = 4 } },
    };

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 5));
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 6));
}

test "formal Z3 adapter totalizes signed min divided by negative one" {
    const max_u256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const min_i256 = "57896044618658097711785492504343953926634992332820282019728792003956564819968";
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = min_i256, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = max_u256, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .div, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .mod, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = min_i256, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 4 } },
        .{ .binary = .{ .op = .eq, .lhs = 3, .rhs = 5 } },
    };

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 6));
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 7));
}

test "formal Z3 adapter signed remainder sign follows dividend" {
    const max_u256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const neg_ten = "115792089237316195423570985008687907853269984665640564039457584007913129639926";
    const neg_three = "115792089237316195423570985008687907853269984665640564039457584007913129639933";
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = neg_ten, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "3", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .mod, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = max_u256, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 3 } },
        .{ .int_lit = .{ .value = "10", .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = neg_three, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .mod, .lhs = 5, .rhs = 6, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "1", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .eq, .lhs = 7, .rhs = 8 } },
    };

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 4));
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 9));
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

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "import Ora.Obligation.Theorems"));
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
    };
    const source_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 0 } }},
    };
    const destination_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 1 } }},
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
                .amount = .{ .term = 0 },
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
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "source := some { root := \"balances\", region := .storage, fields := [], keys := [.parameter { file_id := 0, pattern_id := 0 }] }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "destination := some { root := \"balances\", region := .storage, fields := [], keys := [.parameter { file_id := 0, pattern_id := 1 }] }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "amount := some (.term 0), property := .conservation"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf"));
}

test "formal Lean support accepts u256 resource model properties" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 2 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
    };
    const source_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 0 } }},
    };
    const destination_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 1 } }},
    };
    const properties = [_]obligation.ResourceProperty{
        .amount_non_negative,
        .source_sufficient,
        .destination_no_overflow,
        .same_place_identity,
        .conservation,
    };
    var obligations: [properties.len]obligation.Obligation = undefined;
    var queries: [properties.len]obligation.VerificationQuery = undefined;
    var obligation_ids: [properties.len][1]obligation.Id = undefined;
    for (properties, 0..) |property, index| {
        const id: obligation.Id = @intCast(index + 1);
        obligations[index] = .{
            .id = id,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .resource_op = .{ .op = .move, .domain = "TokenUnit", .ordinal = @intCast(index) } },
            .kind = .{ .resource = .{
                .op = .move,
                .domain = "TokenUnit",
                .source = source_place,
                .destination = destination_place,
                .amount = .{ .term = 0 },
                .property = property,
            } },
        };
        obligation_ids[index] = .{id};
        queries[index] = .{
            .id = @intCast(index + 10),
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .source,
            .kind = .obligation,
            .obligation_ids = &obligation_ids[index],
        };
    }
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    for (queries) |query| {
        switch (obligation_to_lean.querySemanticSupport(set, query)) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }
}

test "formal Lean support treats move resource guards as self-move aware" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 2 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
    };
    const source_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 0 } }},
    };
    const destination_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 1 } }},
    };

    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .move,
        .domain = "TokenUnit",
        .source = source_place,
        .amount = .{ .term = 0 },
        .property = .source_sufficient,
    }, false);
    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .move,
        .domain = "TokenUnit",
        .destination = destination_place,
        .amount = .{ .term = 0 },
        .property = .destination_no_overflow,
    }, false);
    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .destroy,
        .domain = "TokenUnit",
        .source = source_place,
        .amount = .{ .term = 0 },
        .property = .source_sufficient,
    }, true);
    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .create,
        .domain = "TokenUnit",
        .destination = destination_place,
        .amount = .{ .term = 0 },
        .property = .destination_no_overflow,
    }, true);
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
    try expectTypeRefBuiltin(outer.binder.ty, .u256);
    const inner = try expectQuantifiedTerm(result.set, outer.body, .forall, "same");
    try expectTypeRefBuiltin(inner.binder.ty, .u256);
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
    try expectTypeRefBuiltin(free.ty, .u256);

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "x");
    try expectTypeRefBuiltin(outer.binder.ty, .u256);
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
        \\    %ens = ora.cmp "sle", %arg0, %arg0 : !ora.int<64, true>, !ora.int<64, true> -> i1
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
                try expectTypeRefBuiltin(free.ty, .u32);
                const rhs_free = try expectFreeVarTermRef(result.set, binary.rhs, 11, 2, "uy");
                try expectTypeRefBuiltin(rhs_free.ty, .u32);
            },
            .assume => {
                const free = try expectFreeVarTermRef(result.set, binary.lhs, 11, 3, "who");
                try expectTypeRefBuiltin(free.ty, .address);
            },
            else => return error.TestUnexpectedResult,
        }
    }

    const sx = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "sx");
    try expectTypeRefBuiltin(sx.binder.ty, .i64);
    const uy = try expectQuantifiedTerm(result.set, sx.body, .forall, "uy");
    try expectTypeRefBuiltin(uy.binder.ty, .u32);
    const who = try expectQuantifiedTerm(result.set, uy.body, .forall, "who");
    try expectTypeRefBuiltin(who.binder.ty, .address);
}

test "formal obligation MLIR adapter projects signed i256 comparison tags" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @signed_cmp(%arg0: !ora.int<256, true>, %arg1: !ora.int<256, true>) attributes {
        \\    ora.param_names = ["a", "b"],
        \\    ora.param_binding_ids = ["file:21:pattern:1", "file:21:pattern:2"]
        \\  } {
        \\    %cmp = ora.cmp "slt", %arg0, %arg1 : !ora.int<256, true>, !ora.int<256, true> -> i1
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

    const a = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "a");
    try expectTypeRefBuiltin(a.binder.ty, .i256);
    const b = try expectQuantifiedTerm(result.set, a.body, .forall, "b");
    try expectTypeRefBuiltin(b.binder.ty, .i256);
    const body = try expectBinaryTerm(result.set, b.body, .slt);
    try expectTypeRefBuiltin(body.ty, .i256);
    try expectBoundVarTerm(result.set, body.lhs, 1, "a");
    try expectBoundVarTerm(result.set, body.rhs, 0, "b");
    try expectEnsuresQuerySupported(result.set);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".slt"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".compilerTypeId 12"));
}

test "formal obligation Lean support rejects narrow signed comparison width" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @signed_narrow(%arg0: !ora.int<64, true>, %arg1: !ora.int<64, true>) attributes {
        \\    ora.param_names = ["a", "b"],
        \\    ora.param_binding_ids = ["file:22:pattern:1", "file:22:pattern:2"]
        \\  } {
        \\    %cmp = ora.cmp "slt", %arg0, %arg1 : !ora.int<64, true>, !ora.int<64, true> -> i1
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
    const a = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "a");
    try expectTypeRefBuiltin(a.binder.ty, .i64);
    const b = try expectQuantifiedTerm(result.set, a.body, .forall, "b");
    const body = try expectBinaryTerm(result.set, b.body, .slt);
    try expectTypeRefBuiltin(body.ty, .i64);
    try expectEnsuresQueryUnsupported(result.set, .unsupported_comparison_width);
}

test "formal obligation MLIR adapter blocks signed predicate on unsigned operands" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad_signedness(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["a", "b"],
        \\    ora.param_binding_ids = ["file:23:pattern:1", "file:23:pattern:2"]
        \\  } {
        \\    %cmp = ora.cmp "slt", %arg0, %arg1 : !ora.int<256, false>, !ora.int<256, false> -> i1
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
    try testing.expectEqual(obligation.DiagnosticKind.comparison_signedness_mismatch, result.set.diagnostics[0].kind);
    try testing.expect(std.mem.containsAtLeast(u8, result.set.diagnostics[0].message, 1, "predicate signedness"));

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"op\":\"slt\""));
}

test "formal obligation Lean signed comparison support requires compiler type identity" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 31, .pattern_id = 1 }, .name = "a", .ty = .{ .spelling = "i256" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 31, .pattern_id = 2 }, .name = "b", .ty = .{ .spelling = "i256" } } } },
        .{ .binary = .{ .op = .slt, .lhs = 0, .rhs = 1, .ty = .{ .spelling = "i256" } } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "signed_cmp" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "signed_cmp" } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "signed_cmp" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .logical_role = .ensures,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectEnsuresQueryUnsupported(set, .unknown_signedness);
}

test "formal obligation Lean supports signed result, storage, and old operands" {
    const terms = [_]obligation.Term{
        .result,
        .{ .place_read = .{ .root = "reserve", .region = .storage } },
        .{ .old = 1 },
        .{ .binary = .{ .op = .sle, .lhs = 1, .rhs = 2, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .sge, .lhs = 0, .rhs = 2, .ty = builtinTypeRef(.i256) } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "signed_storage" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "signed_storage", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 3 },
            } },
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "signed_result" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "signed_result", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .invariant,
                .formula = .{ .term = 4 },
            } },
        },
    };
    const ensures_obligation_ids = [_]obligation.Id{1};
    const invariant_obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "signed_storage" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .logical_role = .ensures,
            .obligation_ids = &ensures_obligation_ids,
        },
        .{
            .id = 4,
            .owner = .{ .function = .{ .name = "signed_result" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .logical_role = .invariant,
            .obligation_ids = &invariant_obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectEnsuresQuerySupported(set);
    try expectLogicalQuerySupported(set, .invariant);
}

test "formal obligation MLIR adapter projects div and rem value terms with opcode signedness" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @div_rem_terms(%arg0: i256, %arg1: i256) attributes {
        \\    ora.param_names = ["x", "y"],
        \\    ora.param_binding_ids = ["file:31:pattern:1", "file:31:pattern:2"]
        \\  } {
        \\    %udiv = arith.divui %arg0, %arg1 : i256
        \\    %urem = arith.remui %arg0, %arg1 : i256
        \\    %sdiv = arith.divsi %arg0, %arg1 : i256
        \\    %srem = arith.remsi %arg0, %arg1 : i256
        \\    %ucmp = arith.cmpi eq, %udiv, %urem : i256
        \\    "ora.requires"(%ucmp) : (i1) -> ()
        \\    %scmp = arith.cmpi eq, %sdiv, %srem : i256
        \\    "ora.ensures"(%scmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQuerySupported(result.set);

    var unsigned_divs: usize = 0;
    var unsigned_mods: usize = 0;
    var signed_divs: usize = 0;
    var signed_mods: usize = 0;
    for (result.set.terms) |term| {
        if (term != .binary) continue;
        const binary = term.binary;
        switch (binary.op) {
            .div => {
                if (binary.ty) |ty| {
                    if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.u256).comptime_type_id) {
                        unsigned_divs += 1;
                    } else if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.i256).comptime_type_id) {
                        signed_divs += 1;
                    }
                }
            },
            .mod => {
                if (binary.ty) |ty| {
                    if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.u256).comptime_type_id) {
                        unsigned_mods += 1;
                    } else if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.i256).comptime_type_id) {
                        signed_mods += 1;
                    }
                }
            },
            else => {},
        }
    }
    try testing.expect(unsigned_divs >= 1);
    try testing.expect(unsigned_mods >= 1);
    try testing.expect(signed_divs >= 1);
    try testing.expect(signed_mods >= 1);

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".div"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".mod_"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".compilerTypeId 6"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".compilerTypeId 12"));
}

test "formal obligation Lean rejects narrow div and rem arithmetic width" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @narrow_div(%arg0: i32, %arg1: i32) attributes {
        \\    ora.param_names = ["x", "y"],
        \\    ora.param_binding_ids = ["file:32:pattern:1", "file:32:pattern:2"]
        \\  } {
        \\    %q = arith.divui %arg0, %arg1 : i32
        \\    %cmp = arith.cmpi eq, %q, %q : i32
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
    try expectEnsuresQueryUnsupported(result.set, .unsupported_arithmetic_width);
}

test "formal obligation Lean rejects signed comparison over unsigned div result" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @mixed_div_signedness(%arg0: i256, %arg1: i256) attributes {
        \\    ora.param_names = ["x", "y"],
        \\    ora.param_binding_ids = ["file:33:pattern:1", "file:33:pattern:2"]
        \\  } {
        \\    %q = arith.divui %arg0, %arg1 : i256
        \\    %cmp = arith.cmpi slt, %q, %q : i256
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
    try expectEnsuresQueryUnsupported(result.set, .mixed_arithmetic_signedness);
}

test "formal obligation MLIR adapter projects u256 scalar sload as place_read term" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @check() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .le);
    try expectPlaceReadTerm(result.set, body.lhs, "reserve", .storage);
    try expectPlaceReadTerm(result.set, body.rhs, "reserve", .storage);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":1,\"origin_value\":0,\"total\":1,\"term_ratio_basis_points\":10000}"));

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".placeRead { root := \"reserve\", region := .storage"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "def emittedQuery_"));
}

test "formal obligation MLIR adapter projects signed i256 scalar sload for signed comparison" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, true>
        \\    func.func @check() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, true>
        \\      %cmp = ora.cmp "sle", %reserve, %reserve : !ora.int<256, true>, !ora.int<256, true> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .sle);
    try expectTypeRefBuiltin(body.ty, .i256);
    try expectPlaceReadTerm(result.set, body.lhs, "reserve", .storage);
    try expectPlaceReadTerm(result.set, body.rhs, "reserve", .storage);
    try expectEnsuresQuerySupported(result.set);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".sle"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".compilerTypeId 12"));
}

test "formal obligation MLIR adapter requires explicit write set before scalar sload projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @check() {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":0,\"origin_value\":1,\"total\":1,\"term_ratio_basis_points\":0}"));
}

test "formal obligation MLIR adapter does not project scalar sload when function writes root" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balance" : !ora.int<256, false>
        \\    func.func @mutates(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["k", "next"],
        \\      ora.param_binding_ids = ["file:17:pattern:1", "file:17:pattern:2"],
        \\      ora.write_slots = ["balance"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %before = ora.sload "balance" : !ora.int<256, false>
        \\      %req = ora.cmp "eq", %before, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.requires"(%req) : (i1) -> ()
        \\      ora.sstore %arg1, "balance" : !ora.int<256, false>
        \\      %after = ora.sload "balance" : !ora.int<256, false>
        \\      %ens = ora.cmp "eq", %after, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%ens) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":0,\"origin_value\":1,\"total\":1,\"term_ratio_basis_points\":0}"));
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter does not project scalar sload in functions with external calls" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @calls(%target: !ora.address, %gas: !ora.int<256, false>, %calldata: !ora.bytes) attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %success, %returndata = ora.external_call %target, %gas, %calldata {call_kind = "call", method_name = "ping", trait_name = "Remote"} : !ora.address, !ora.int<256, false>, !ora.bytes -> i1, !ora.bytes
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter keeps non-u256 scalar sload outside Lean term fragment" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "small" : !ora.int<32, false>
        \\    func.func @check() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %small = ora.sload "small" : !ora.int<32, false>
        \\      %cmp = ora.cmp "ule", %small, %small : !ora.int<32, false>, !ora.int<32, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    for (result.set.obligations) |item| {
        if (item.kind != .logical or item.kind.logical.role != .ensures) continue;
        try testing.expect(item.kind.logical.formula == .origin_value);
    }

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":0,\"origin_value\":1,\"total\":1,\"term_ratio_basis_points\":0}"));
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));

    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
}

test "formal obligation MLIR adapter projects old written scalar sload as entry place read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @snapshot() attributes {
        \\      ora.write_slots = ["counter"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %counter = ora.sload "counter" : !ora.int<256, false>
        \\      %old = ora.old %counter : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .le);
    try expectOldPlaceReadTerm(result.set, body.lhs, "counter", .storage);
    try expectOldPlaceReadTerm(result.set, body.rhs, "counter", .storage);
    try testing.expectEqual(@as(usize, 2), countOldTerms(result.set));

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".old "));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".placeRead { root := \"counter\", region := .storage"));
}

test "formal obligation MLIR adapter collapses old unwritten scalar sload to stable place read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balance" : !ora.int<256, false>
        \\    func.func @readonly() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balance = ora.sload "balance" : !ora.int<256, false>
        \\      %old = ora.old %balance : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .eq);
    try expectPlaceReadTerm(result.set, body.lhs, "balance", .storage);
    try expectPlaceReadTerm(result.set, body.rhs, "balance", .storage);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, ".old ") == null);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".placeRead { root := \"balance\", region := .storage"));
}

test "formal obligation MLIR adapter does not collapse old with bare written-root read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balance" : !ora.int<256, false>
        \\    func.func @mutates() attributes {
        \\      ora.write_slots = ["balance"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balance = ora.sload "balance" : !ora.int<256, false>
        \\      %old = ora.old %balance : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter requires write metadata before old scalar sload projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @snapshot() {
        \\      %counter = ora.sload "counter" : !ora.int<256, false>
        \\      %old = ora.old %counter : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter requires complete write metadata before scalar sload projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @check() attributes {
        \\      ora.write_slots = []
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter does not project old scalar sload in functions with external calls" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @snapshot(%target: !ora.address, %gas: !ora.int<256, false>, %calldata: !ora.bytes) attributes {
        \\      ora.write_slots = ["counter"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %counter = ora.sload "counter" : !ora.int<256, false>
        \\      %old = ora.old %counter : !ora.int<256, false> -> !ora.int<256, false>
        \\      %success, %returndata = ora.external_call %target, %gas, %calldata {call_kind = "call", method_name = "ping", trait_name = "Remote"} : !ora.address, !ora.int<256, false>, !ora.bytes -> i1, !ora.bytes
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter keeps arbitrary old expression outside Lean term fragment" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @old_expr(%arg0: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["x"],
        \\    ora.param_binding_ids = ["file:23:pattern:1"],
        \\    ora.write_slots = [],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %sum = ora.add_wrapping %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> !ora.int<256, false>
        \\    %old = ora.old %sum : !ora.int<256, false> -> !ora.int<256, false>
        \\    %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
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
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter supports old written scalar sload in invariant formulas" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "total" : !ora.int<256, false>
        \\    func.func @loop_snapshot() attributes {
        \\      ora.write_slots = ["total"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %total = ora.sload "total" : !ora.int<256, false>
        \\      %old = ora.old %total : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.invariant"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .invariant));
    try expectLogicalQuerySupported(result.set, .invariant);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .invariant), .le);
    try expectOldPlaceReadTerm(result.set, body.lhs, "total", .storage);
    try expectOldPlaceReadTerm(result.set, body.rhs, "total", .storage);
    try testing.expectEqual(@as(usize, 2), countOldTerms(result.set));
}

test "formal obligation MLIR adapter rejects invariant with old and bare written-root read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "total" : !ora.int<256, false>
        \\    func.func @loop_snapshot() attributes {
        \\      ora.write_slots = ["total"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %total = ora.sload "total" : !ora.int<256, false>
        \\      %old = ora.old %total : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %old, %total : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.invariant"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .invariant));
    try expectLogicalQueryUnsupported(result.set, .invariant, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 1), countOldTerms(result.set));
}

test "formal obligation MLIR adapter projects read-only keyed map sload as place_read term" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:41:pattern:1"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const quantified = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "owner");
    const body = try expectBinaryTerm(result.set, quantified.body, .eq);
    try expectPlaceReadTermWithParameterKeyCount(result.set, body.lhs, "balances", 1);
    try expectPlaceReadTermWithParameterKeyCount(result.set, body.rhs, "balances", 1);

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "root := \"balances\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "keys := [.parameter { file_id := "));
}

test "formal obligation MLIR adapter ties parameter place keys to free variable ids" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["user", "other"],
        \\      ora.param_binding_ids = ["file:77:pattern:1", "file:77:pattern:2"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %req = ora.cmp "ule", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.requires"(%req) : (i1) -> ()
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %user) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    var user_var_id: ?obligation.FreeVarId = null;
    var other_var_id: ?obligation.FreeVarId = null;
    var place_key_id: ?obligation.FreeVarId = null;
    for (result.set.terms) |term| {
        switch (term) {
            .variable => |variable| switch (variable) {
                .free => |free| {
                    if (std.mem.eql(u8, free.name, "user")) user_var_id = free.id;
                    if (std.mem.eql(u8, free.name, "other")) other_var_id = free.id;
                },
                .bound => {},
            },
            .place_read => |place| {
                if (std.mem.eql(u8, place.root, "balances")) {
                    try testing.expectEqual(@as(usize, 1), place.keys.len);
                    try testing.expect(place.keys[0] == .parameter);
                    place_key_id = place.keys[0].parameter;
                }
            },
            else => {},
        }
    }

    const user_id = user_var_id orelse return error.TestUnexpectedResult;
    const other_id = other_var_id orelse return error.TestUnexpectedResult;
    const key_id = place_key_id orelse return error.TestUnexpectedResult;
    try testing.expect(obligation.freeVarIdEql(user_id, key_id));
    try testing.expect(!obligation.freeVarIdEql(other_id, key_id));

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".free { id := { file_id := 77, pattern_id := 1 }, name := \"user\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "keys := [.parameter { file_id := 77, pattern_id := 1 }]"));
}

test "formal obligation source collector rejects loop block argument map keys" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    pub fn check(x: u256, cap: u256)
        \\        ensures balances[x] <= cap
        \\    {
        \\        var i: u256 = 0;
        \\        while (i < x)
        \\            invariant balances[i] <= cap
        \\        {
        \\            i = i + 1;
        \\        }
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @check"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.invariant"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .invariant));
    try expectEnsuresQuerySupported(result.set);
    try expectLogicalQueryUnsupported(result.set, .invariant, .unsupported_origin_value);
}

test "formal obligation source collector projects environment map keys" {
    const source_text =
        \\contract EnvStorageProjection {
        \\    storage balances: map<address, u256>;
        \\    storage origins: map<address, u256>;
        \\
        \\    pub fn check()
        \\        ensures balances[std.msg.sender()] == balances[std.msg.sender()]
        \\        ensures origins[std.tx.origin()] == origins[std.tx.origin()]
        \\    {
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.evm.caller"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.evm.origin"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .ensures));

    var sender_reads: usize = 0;
    var origin_reads: usize = 0;
    for (result.set.terms) |term| {
        if (term != .place_read) continue;
        if (std.mem.eql(u8, term.place_read.root, "balances")) {
            try testing.expectEqual(@as(usize, 1), term.place_read.keys.len);
            try testing.expect(term.place_read.keys[0] == .msg_sender);
            sender_reads += 1;
        } else if (std.mem.eql(u8, term.place_read.root, "origins")) {
            try testing.expectEqual(@as(usize, 1), term.place_read.keys.len);
            try testing.expect(term.place_read.keys[0] == .tx_origin);
            origin_reads += 1;
        }
    }
    try testing.expectEqual(@as(usize, 2), sender_reads);
    try testing.expectEqual(@as(usize, 2), origin_reads);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".msgSender"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".txOrigin"));
}

test "formal obligation source collector projects statically disjoint constant map paths" {
    const source_text =
        \\contract StorageProjection {
        \\    storage buckets: map<u256, u256>;
        \\
        \\    pub fn check()
        \\        ensures buckets[2] == old(buckets[2])
        \\    {
        \\        buckets[1] = 7;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots = [\"buckets[1]\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .eq);
    try expectPlaceReadTermWithConstantKeys(result.set, body.lhs, "buckets", &.{"2"});
    try expectPlaceReadTermWithConstantKeys(result.set, body.rhs, "buckets", &.{"2"});
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":1,\"origin_value\":0,\"total\":1,\"term_ratio_basis_points\":10000}"));

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 2, "root := \"buckets\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 2, "keys := [.constant \"2\"]"));
    try testing.expect(std.mem.indexOf(u8, lean, ".old ") == null);
}

test "formal obligation source collector keeps conditional map post-state equality outside Lean projection" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    pub fn check(user: u256, other: u256, value: u256)
        \\        requires user != other
        \\        ensures balances[other] == old(balances[other])
        \\    {
        \\        balances[user] = value;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots = [\"balances[param#0]\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));
    try expectEffectFrameQuerySupported(result.set, .read_preserved_by_key_evidence);
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
}

test "formal obligation source queries do not mix same-ordinal parameter owners" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    pub fn first(x: u256, cap: u256)
        \\        requires x <= cap
        \\        ensures balances[x] == old(balances[x])
        \\    {
        \\    }
        \\
        \\    pub fn second(x: u256, cap: u256)
        \\        requires x <= cap
        \\        ensures balances[x] == old(balances[x])
        \\    {
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @first"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @second"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "ora.param_binding_ids"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .ensures));
    try expectQueriesOwnerScoped(result.set);
}

test "formal obligation source collector does not split msg.sender from tx.origin" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn check(value: u256)
        \\        ensures balances[std.tx.origin()] == old(balances[std.tx.origin()])
        \\    {
        \\        balances[std.msg.sender()] = value;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots = [\"balances[msg.sender]\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
}

test "formal obligation MLIR adapter requires complete write metadata before keyed map projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:42:pattern:1"],
        \\      ora.write_slots = []
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter blocks parameter effect paths without binding ids" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%owner: !ora.int<256, false>) attributes {
        \\      ora.write_slots = ["balances[param#0]"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(obligation.ArtifactBlockReason.blocking_diagnostic, result.set.artifactDecision().blocked);
    var saw_missing_binding = false;
    for (result.set.diagnostics) |diagnostic| {
        if (diagnostic.kind != .missing_effect_path) continue;
        if (std.mem.containsAtLeast(
            u8,
            diagnostic.message,
            1,
            "references parameter 0 but ora.param_binding_ids is missing or too short",
        )) {
            saw_missing_binding = true;
        }
    }
    try testing.expect(saw_missing_binding);
}

test "formal obligation MLIR adapter does not project keyed map read when function writes root" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @mutates(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:43:pattern:1"],
        \\      ora.write_slots = ["balances[param#0]"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter blocks prefix map path projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\    func.func @mutates(%owner: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner", "spender"],
        \\      ora.param_binding_ids = ["file:143:pattern:1", "file:143:pattern:2"],
        \\      ora.write_slots = ["allowances[param#0]"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %allowances = ora.sload "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\      %owner_allowances = "ora.map_get"(%allowances, %owner) : (!ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>, !ora.int<256, false>) -> !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %allowance = "ora.map_get"(%owner_allowances, %spender) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %old = ora.old %allowance : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %allowance, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation source collector does not project map read when helper writes root" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    fn overwrite(owner: u256, value: u256) {
        \\        balances[owner] = value;
        \\    }
        \\
        \\    pub fn check(owner: u256)
        \\        ensures balances[owner] == balances[owner]
        \\    {
        \\        overwrite(owner, 1);
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @check"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"balances"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter emits read preserved frame for static constant disjoint paths" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface() attributes {
        \\    ora.effect = "readwrites",
        \\    ora.modifies_slots = ["buckets[1]"],
        \\    ora.read_slots = ["buckets[2]"],
        \\    ora.write_slots = ["buckets[1]"],
        \\    ora.write_slots_complete = true
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
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_frame));

    var saw_read_preserved = false;
    for (result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_frame) continue;
        saw_read_preserved = true;
        const frame = item.kind.effect_frame;
        try testing.expectEqual(@as(usize, 1), frame.declared.len);
        try testing.expectEqual(@as(usize, 1), frame.actual.len);
        try testing.expectEqualStrings("buckets", frame.declared[0].root);
        try expectConstantKey(frame.declared[0].keys[0], "1");
        try testing.expectEqualStrings("buckets", frame.actual[0].root);
        try expectConstantKey(frame.actual[0].keys[0], "2");
    }
    try testing.expect(saw_read_preserved);
}

test "formal obligation MLIR adapter emits evidence-backed frame for parameter disequality" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:501:pattern:1", "file:501:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));

    var evidence_obligation_id: ?obligation.Id = null;
    var saw_evidence = false;
    for (result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        saw_evidence = true;
        evidence_obligation_id = item.id;
        const frame = item.kind.effect_frame;
        try testing.expectEqual(@as(usize, 1), frame.declared.len);
        try testing.expectEqual(@as(usize, 1), frame.actual.len);
        try testing.expectEqual(@as(usize, 1), frame.evidence.len);
        try testing.expectEqual(result.set.assumptions[0].id, frame.evidence[0].assumption_id);
        try testing.expectEqual(@as(u32, 0), frame.evidence[0].key_index);
        try testing.expect(obligation.freeVarIdEql(.{ .file_id = 501, .pattern_id = 1 }, frame.evidence[0].lhs));
        try testing.expect(obligation.freeVarIdEql(.{ .file_id = 501, .pattern_id = 2 }, frame.evidence[0].rhs));
        try expectParameterKey(frame.evidence[0].write.keys[0], .{ .file_id = 501, .pattern_id = 1 });
        try expectParameterKey(frame.evidence[0].read.keys[0], .{ .file_id = 501, .pattern_id = 2 });
    }
    try testing.expect(saw_evidence);

    var saw_query_with_assumption = false;
    for (result.set.queries) |query| {
        if (query.obligation_ids.len != 1 or query.obligation_ids[0] != evidence_obligation_id.?) continue;
        saw_query_with_assumption = true;
        try testing.expectEqual(@as(usize, 1), query.assumption_ids.len);
        try testing.expectEqual(result.set.assumptions[0].id, query.assumption_ids[0]);
        switch (obligation_to_lean.querySemanticSupport(result.set, query)) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }
    try testing.expect(saw_query_with_assumption);
    try expectEffectFrameQuerySupported(result.set, .read_preserved_by_key_evidence);

    const dump = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(dump);
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"schema_version\":4"));
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"relation\":\"read_preserved_by_key_evidence\""));
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"evidence\":[{\"kind\":\"free_var_disequality\""));
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"key_index\":0"));
}

test "formal obligation MLIR adapter accepts symmetric parameter disequality evidence" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:502:pattern:1", "file:502:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %other, %user : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter requires complete write slots for key evidence frame" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:504:pattern:1", "file:504:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"]
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter requires evidence for every non-static read write pair" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>, %admin: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other", "admin"],
        \\    ora.param_binding_ids = ["file:503:pattern:1", "file:503:pattern:2", "file:503:pattern:3"],
        \\    ora.modifies_slots = ["balances[param#0]", "balances[param#2]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]", "balances[param#2]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter rejects same-key evidence-backed frame" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:505:pattern:1", "file:505:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#0]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter rejects prefix-path evidence-backed frame" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%owner: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["owner", "spender"],
        \\    ora.param_binding_ids = ["file:506:pattern:1", "file:506:pattern:2"],
        \\    ora.modifies_slots = ["allowances[param#0]"],
        \\    ora.read_slots = ["allowances[param#0][param#1]"],
        \\    ora.write_slots = ["allowances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %owner, %spender : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter does not borrow key evidence across owners" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @provider(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:507:pattern:1", "file:507:pattern:2"]
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:507:pattern:1", "file:507:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
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
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
    try expectQueriesOwnerScoped(result.set);
}

test "formal obligation MLIR adapter emits evidence for nested differing parameter key" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%owner: !ora.int<256, false>, %other: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["owner", "other", "spender"],
        \\    ora.param_binding_ids = ["file:508:pattern:1", "file:508:pattern:2", "file:508:pattern:3"],
        \\    ora.modifies_slots = ["allowances[param#0][param#2]"],
        \\    ora.read_slots = ["allowances[param#1][param#2]"],
        \\    ora.write_slots = ["allowances[param#0][param#2]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %owner, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));

    const query = try findEffectFrameQuery(result.set, .read_preserved_by_key_evidence);
    try testing.expectEqual(@as(usize, 1), query.assumption_ids.len);
    for (result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        try testing.expectEqual(@as(usize, 1), item.kind.effect_frame.evidence.len);
        try testing.expectEqual(@as(u32, 0), item.kind.effect_frame.evidence[0].key_index);
    }
}

test "formal obligation Lean support rejects unsupported key evidence formulas" {
    const owner: obligation.Owner = .{ .function = .{ .name = "effect_surface" } };
    const lhs_id: obligation.FreeVarId = .{ .file_id = 601, .pattern_id = 1 };
    const rhs_id: obligation.FreeVarId = .{ .file_id = 601, .pattern_id = 2 };
    const read_keys = [_]obligation.PlaceKey{.{ .parameter = rhs_id }};
    const write_keys = [_]obligation.PlaceKey{.{ .parameter = lhs_id }};
    const read: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &read_keys };
    const write: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &write_keys };
    const assumption_ids = [_]obligation.Id{10};
    const u256_ty: obligation.TypeRef = .{ .spelling = "u256" };
    const user: obligation.Term = .{ .variable = .{ .free = .{ .id = lhs_id, .name = "user", .ty = u256_ty } } };
    const other: obligation.Term = .{ .variable = .{ .free = .{ .id = rhs_id, .name = "other", .ty = u256_ty } } };

    {
        const terms = [_]obligation.Term{
            user,
            other,
            .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
            .{ .unary = .{ .op = .not, .operand = 2 } },
        };
        try expectSyntheticKeyEvidenceUnsupported(
            &terms,
            3,
            .requires,
            &assumption_ids,
            owner,
            owner,
            read,
            write,
            lhs_id,
            rhs_id,
            .unsupported_key_disjoint_evidence_formula,
        );
    }

    {
        const terms = [_]obligation.Term{
            user,
            .{ .int_lit = .{ .value = "42", .ty = u256_ty } },
            .{ .binary = .{ .op = .ne, .lhs = 0, .rhs = 1 } },
        };
        try expectSyntheticKeyEvidenceUnsupported(
            &terms,
            2,
            .requires,
            &assumption_ids,
            owner,
            owner,
            read,
            write,
            lhs_id,
            rhs_id,
            .unsupported_key_disjoint_evidence_formula,
        );
    }

    {
        const terms = [_]obligation.Term{
            user,
            other,
            .{ .binary = .{ .op = .ne, .lhs = 0, .rhs = 1 } },
        };
        try expectSyntheticKeyEvidenceUnsupported(
            &terms,
            2,
            .assume,
            &assumption_ids,
            owner,
            owner,
            read,
            write,
            lhs_id,
            rhs_id,
            .unsupported_key_disjoint_evidence_formula,
        );
    }
}

test "formal obligation Lean support rejects non-u256 key evidence variables" {
    const owner: obligation.Owner = .{ .function = .{ .name = "effect_surface" } };
    const lhs_id: obligation.FreeVarId = .{ .file_id = 602, .pattern_id = 1 };
    const rhs_id: obligation.FreeVarId = .{ .file_id = 602, .pattern_id = 2 };
    const read_keys = [_]obligation.PlaceKey{.{ .parameter = rhs_id }};
    const write_keys = [_]obligation.PlaceKey{.{ .parameter = lhs_id }};
    const read: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &read_keys };
    const write: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &write_keys };
    const address_ty: obligation.TypeRef = .{ .spelling = "address" };
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = lhs_id, .name = "user", .ty = address_ty } } },
        .{ .variable = .{ .free = .{ .id = rhs_id, .name = "other", .ty = address_ty } } },
        .{ .binary = .{ .op = .ne, .lhs = 0, .rhs = 1 } },
    };

    try expectSyntheticKeyEvidenceUnsupported(
        &terms,
        2,
        .requires,
        &.{},
        owner,
        owner,
        read,
        write,
        lhs_id,
        rhs_id,
        .key_disjoint_evidence_type_unsupported,
    );
}

test "formal obligation MLIR adapter does not project keyed map read with unknown key" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @unknown_key(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:44:pattern:1"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %key = ora.add_wrapping %owner, %owner : !ora.int<256, false>, !ora.int<256, false> -> !ora.int<256, false>
        \\      %balance = "ora.map_get"(%balances, %key) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
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
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter projects nested read-only map paths as distinct roots" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    ora.global "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\    func.func @check(%owner: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner", "spender"],
        \\      ora.param_binding_ids = ["file:45:pattern:1", "file:45:pattern:2"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %allowances = ora.sload "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\      %owner_allowances = "ora.map_get"(%allowances, %owner) : (!ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>, !ora.int<256, false>) -> !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %allowance = "ora.map_get"(%owner_allowances, %spender) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %balance_cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      %allowance_cmp = ora.cmp "eq", %allowance, %allowance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%balance_cmp) : (i1) -> ()
        \\      "ora.ensures"(%allowance_cmp) : (i1) -> ()
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
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .ensures));

    var saw_balances = false;
    var saw_allowances = false;
    for (result.set.terms) |term| {
        if (term != .place_read) continue;
        if (std.mem.eql(u8, term.place_read.root, "balances")) {
            saw_balances = true;
            try testing.expectEqual(@as(usize, 1), term.place_read.keys.len);
            try expectParameterKey(term.place_read.keys[0], .{ .file_id = 45, .pattern_id = 1 });
        } else if (std.mem.eql(u8, term.place_read.root, "allowances")) {
            saw_allowances = true;
            try testing.expectEqual(@as(usize, 2), term.place_read.keys.len);
            try expectParameterKey(term.place_read.keys[0], .{ .file_id = 45, .pattern_id = 1 });
            try expectParameterKey(term.place_read.keys[1], .{ .file_id = 45, .pattern_id = 2 });
        }
    }
    try testing.expect(saw_balances);
    try testing.expect(saw_allowances);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "root := \"balances\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "root := \"allowances\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "keys := [.parameter { file_id := 45, pattern_id := 1 }, .parameter { file_id := 45, pattern_id := 2 }]"));
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
        \\  func.func @effect_surface(%account: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["account"],
        \\    ora.param_binding_ids = ["file:88:pattern:1"],
        \\    ora.modifies_slots = ["balances[param#0]", "config.owner"],
        \\    ora.read_slots = ["balances[param#0]", "transient:scratch"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
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
    try expectParameterKey(writes.declared[0].keys[0], .{ .file_id = 88, .pattern_id = 1 });
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
        \\  func.func @transfer_resource(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["balances", "from", "to", "amount"],
        \\    ora.param_binding_ids = ["file:201:pattern:0", "file:201:pattern:1", "file:201:pattern:2", "file:201:pattern:3"]
        \\  } {
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
    try testing.expectEqual(@as(usize, 5), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 6), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 5), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .amount_non_negative));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .source_sufficient));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .destination_no_overflow));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .same_place_identity));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .conservation));
    inline for (.{
        obligation.ResourceProperty.amount_non_negative,
        obligation.ResourceProperty.source_sufficient,
        obligation.ResourceProperty.destination_no_overflow,
        obligation.ResourceProperty.same_place_identity,
        obligation.ResourceProperty.conservation,
    }) |property| {
        switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .move, property))) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }

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
        try expectParameterKey(source.keys[0], .{ .file_id = 201, .pattern_id = 1 });
        try expectParameterKey(destination.keys[0], .{ .file_id = 201, .pattern_id = 2 });
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
    try testing.expectEqual(@as(usize, 4), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .create, .destination_no_overflow));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .destroy, .source_sufficient));
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .create, .amount_non_negative))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .create, .destination_no_overflow))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .destroy, .amount_non_negative))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .destroy, .source_sufficient))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }

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
        \\    ora.param_names = ["flag", "balances", "from", "to", "amount", "x", "y"],
        \\    ora.param_binding_ids = ["file:202:pattern:0", "file:202:pattern:1", "file:202:pattern:2", "file:202:pattern:3", "file:202:pattern:4", "file:202:pattern:5", "file:202:pattern:6"],
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
    try testing.expectEqual(@as(usize, 12), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 13), result.set.queries.len);
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
    try testing.expectEqual(@as(usize, 10), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);

    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"artifact_decision\",\"schema_version\":4,\"status\":\"blocked\",\"reason\":\"missing_proof\"}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":4,\"assumptions\":1,\"obligations\":12,\"queries\":13"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_obligation_links\":12"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"obligation_kinds\":{\"logical\":4,\"runtime_guard\":1,\"type_wf\":0,\"type_relation\":0,\"region_relation\":0,\"effect_frame\":1,\"resource\":5,\"quantifier\":1,\"filtered_input\":0,\"backend_fact\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_backends\":{\"unspecified\":13,\"z3\":0,\"lean\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_results\":{\"missing\":13,\"sat\":0,\"unsat\":0,\"unknown\":0,\"proved\":0,\"failed\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"assumption\",\"schema_version\":4"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"logical\",\"role\":\"ensures\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"addition_overflow\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"division_by_zero\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"runtime_guard\",\"guard_id\":\"guard:coverage:flag\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"effect_frame\",\"relation\":\"write_covered_by_modifies\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"resource\",\"op\":\"move\",\"domain\":\"TokenUnit\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"property\":\"conservation\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"quantifier\",\"quantifier\":\"forall\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"query\",\"schema_version\":4"));
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
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":4"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"assumption\",\"schema_version\":4"));
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
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":4"));
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
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":4"));
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
