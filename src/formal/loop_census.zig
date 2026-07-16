//! Deterministic, measurement-only loop census.
//!
//! This module consumes existing AST, semantic-effect, formal-support, and Z3
//! prepared-query analyses. It does not participate in compilation decisions.

const std = @import("std");
const ora_root = @import("ora_root");
const mlir_c_api = @import("mlir_c_api");
const z3_verification = @import("ora_z3_verification");
const obligation = @import("obligation.zig");
const obligation_from_mlir = @import("obligation_from_mlir.zig");
const obligation_to_lean = @import("obligation_to_lean.zig");

const compiler = ora_root.compiler;
const ast = compiler.ast;
const sema = compiler.sema;
const source = compiler.source;
const mlir = mlir_c_api.c;

pub const schema = "ora.loop_census.file.v2";
pub const schema_version: u32 = 2;

const QueryCounts = struct {
    loop_invariant_step: u32 = 0,
    loop_body_safety: u32 = 0,
    loop_invariant_post: u32 = 0,
};

const InvariantSupport = struct {
    line: u32,
    column: u32,
    supported: bool,
    reason: ?[]const u8,
};

const LoopRecord = struct {
    statement_id: ast.StmtId,
    line: u32,
    column: u32,
    function_name: []const u8,
    loop_form: []const u8,
    invariants: []const ast.ExprId,
    is_nested: bool,
    effect_found: bool,
    writes_storage: bool,
    reads_storage: bool,
    has_calls: bool,
    has_body_branch: bool,
    has_resource_operation: bool,
    has_break_or_continue: bool,
    has_error_control_flow: bool,
    has_nested_loop: bool,
    scalar_updates: bool,
    loop_var_types: []const []const u8,
    invariant_support: []const InvariantSupport,
    queries: QueryCounts = .{},
    shape_scalar_fragment: bool,
    shape_excluded_by: []const []const u8,
    scalar_fragment: bool,
    excluded_by: []const []const u8,
};

const RawLoop = struct {
    statement_id: ast.StmtId,
    function_name: []const u8,
    is_nested: bool,
};

const MeasurementError = struct {
    reason: []const u8,
    statement_id: ?u32 = null,
};

const LoopCfgFact = struct {
    statement_index: usize,
    has_body_branch: bool,
};

pub fn writeCompilationJson(
    writer: anytype,
    allocator: std.mem.Allocator,
    requested_path: []const u8,
    compilation: *compiler.driver.Compilation,
) !void {
    const decision = try compilation.artifactEmissionDecision();
    switch (decision) {
        .blocked => |block_reason| {
            const diagnostic = (try firstCompilationDiagnostic(compilation)) orelse blockReasonName(block_reason);
            try writeFailureJson(writer, requested_path, diagnostic);
            return;
        },
        .allowed => |lowering| try writeSuccessfulCompilationJson(writer, allocator, requested_path, compilation, lowering),
    }
}

pub fn writeCompilerErrorJson(writer: anytype, requested_path: []const u8, err: anyerror) !void {
    var buffer: [160]u8 = undefined;
    const diagnostic = std.fmt.bufPrint(&buffer, "compiler_error:{s}", .{@errorName(err)}) catch "compiler_error";
    try writeFailureJson(writer, requested_path, diagnostic);
}

fn writeSuccessfulCompilationJson(
    writer: anytype,
    allocator: std.mem.Allocator,
    requested_path: []const u8,
    compilation: *compiler.driver.Compilation,
    lowering: *const compiler.hir.LoweringResult,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const root_module = compilation.db.sources.module(compilation.root_module_id);
    const file_id = root_module.file_id;
    const ast_file = try compilation.db.astFile(file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    var raw_loops: std.ArrayList(RawLoop) = .empty;
    defer raw_loops.deinit(arena);
    var seen = try arena.alloc(bool, ast_file.statements.len);
    @memset(seen, false);

    for (ast_file.items) |item| {
        switch (item) {
            .Function => |function| {
                var visitor = LoopInventoryVisitor{
                    .allocator = arena,
                    .loops = &raw_loops,
                    .seen = seen,
                    .function_name = function.name,
                };
                try ast.walk.walkBody(LoopInventoryVisitor, &visitor, ast_file, function.body, inventory_walk_options);
            },
            .GhostBlock => |ghost| {
                var visitor = LoopInventoryVisitor{
                    .allocator = arena,
                    .loops = &raw_loops,
                    .seen = seen,
                    .function_name = "<ghost>",
                };
                try ast.walk.walkBody(LoopInventoryVisitor, &visitor, ast_file, ghost.body, inventory_walk_options);
            },
            else => {},
        }
    }

    // Total-accounting fallback for statement-bearing comptime/module bodies
    // that are not owned by a runtime function or ghost block.
    for (ast_file.statements, 0..) |statement, index| {
        switch (statement) {
            .While, .For => {
                if (seen[index]) continue;
                seen[index] = true;
                try raw_loops.append(arena, .{
                    .statement_id = ast.StmtId.fromIndex(index),
                    .function_name = "<module>",
                    .is_nested = false,
                });
            },
            .Switch => |switch_stmt| {
                if (switch_stmt.label == null or seen[index]) continue;
                seen[index] = true;
                try raw_loops.append(arena, .{
                    .statement_id = ast.StmtId.fromIndex(index),
                    .function_name = "<module>",
                    .is_nested = false,
                });
            },
            else => {},
        }
    }
    std.mem.sort(RawLoop, raw_loops.items, {}, lessThanRawLoop);

    var measurement_errors: std.ArrayList(MeasurementError) = .empty;
    defer measurement_errors.deinit(arena);

    var formal_result_opt: ?obligation_from_mlir.CollectResult = obligation_from_mlir.collect(
        allocator,
        lowering.module.raw_module,
        .{},
    ) catch |err| blk: {
        try measurement_errors.append(arena, .{ .reason = try std.fmt.allocPrint(arena, "formal_collection_failed:{s}", .{@errorName(err)}) });
        break :blk null;
    };
    defer if (formal_result_opt) |*result| result.deinit();

    var loop_query_census_opt: ?z3_verification.LoopQueryCensus = blk: {
        var verifier = z3_verification.VerificationPass.init(allocator) catch |err| {
            try measurement_errors.append(arena, .{ .reason = try std.fmt.allocPrint(arena, "query_collector_init_failed:{s}", .{@errorName(err)}) });
            break :blk null;
        };
        defer verifier.deinit();
        break :blk verifier.collectLoopQueryCensus(lowering.module.raw_module) catch |err| {
            try measurement_errors.append(arena, .{ .reason = try std.fmt.allocPrint(arena, "query_collection_failed:{s}", .{@errorName(err)}) });
            break :blk null;
        };
    };
    defer if (loop_query_census_opt) |*census| census.deinit();

    const loop_cfg_facts = try collectLoopCfgFacts(arena, lowering.module.raw_module);

    var records: std.ArrayList(LoopRecord) = .empty;
    defer records.deinit(arena);
    for (raw_loops.items) |raw| {
        try records.append(arena, try buildLoopRecord(
            arena,
            &compilation.db.sources,
            file_id,
            ast_file,
            typecheck,
            if (formal_result_opt) |*result| result.set else null,
            loop_cfg_facts,
            raw,
        ));
    }

    if (loop_query_census_opt) |census| {
        const root_path = compilation.db.sources.file(file_id).path;
        for (census.rows) |query| {
            const statement_index = query.loop_statement_index orelse {
                try measurement_errors.append(arena, .{ .reason = "loop_query_missing_statement_identity" });
                continue;
            };
            if (query.loop_file.len != 0 and !pathsReferToSameSource(root_path, query.loop_file)) continue;
            const record = findRecord(records.items, statement_index) orelse {
                try measurement_errors.append(arena, .{ .reason = "loop_query_owner_not_in_root_ast", .statement_id = statement_index });
                continue;
            };
            switch (query.kind) {
                .LoopInvariantStep => record.queries.loop_invariant_step += 1,
                .LoopBodySafety => record.queries.loop_body_safety += 1,
                .LoopInvariantPost => record.queries.loop_invariant_post += 1,
                else => unreachable,
            }
        }
    }

    try writer.writeAll("{\"schema\":");
    try writeJsonString(writer, schema);
    try writer.print(",\"schema_version\":{d},\"source_file\":", .{schema_version});
    try writeJsonString(writer, requested_path);
    try writer.writeAll(",\"compiling\":true,\"verification_encoding_degraded\":");
    try writer.writeAll(if (loop_query_census_opt) |census| if (census.encoding_degraded) "true" else "false" else "false");
    try writer.writeAll(",\"measurement_errors\":[");
    for (measurement_errors.items, 0..) |entry, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.writeAll("{\"reason\":");
        try writeJsonString(writer, entry.reason);
        if (entry.statement_id) |statement_id| try writer.print(",\"statement_id\":{d}", .{statement_id});
        try writer.writeByte('}');
    }
    try writer.writeAll("],\"loops\":[");
    for (records.items, 0..) |record, index| {
        if (index != 0) try writer.writeByte(',');
        try writeLoopRecord(writer, requested_path, record);
    }
    try writer.writeAll("]}\n");
}

fn buildLoopRecord(
    arena: std.mem.Allocator,
    sources: *const source.SourceStore,
    file_id: source.FileId,
    ast_file: *const ast.AstFile,
    typecheck: *const sema.TypeCheckResult,
    formal_set: ?obligation.ObligationSet,
    loop_cfg_facts: []const LoopCfgFact,
    raw: RawLoop,
) !LoopRecord {
    const statement = ast_file.statement(raw.statement_id).*;
    const LoopData = struct {
        range: source.TextRange,
        form: []const u8,
        invariants: []const ast.ExprId,
        condition: ?ast.ExprId,
        for_stmt: ?ast.ForStmt,
    };
    var loop_bodies: std.ArrayList(ast.BodyId) = .empty;
    defer loop_bodies.deinit(arena);
    const loop_data: LoopData = switch (statement) {
        .While => |loop| blk: {
            try loop_bodies.append(arena, loop.body);
            break :blk .{
                .range = loop.range,
                .form = "while",
                .invariants = loop.invariants,
                .condition = @as(?ast.ExprId, loop.condition),
                .for_stmt = @as(?ast.ForStmt, null),
            };
        },
        .For => |loop| blk: {
            try loop_bodies.append(arena, loop.body);
            break :blk .{
                .range = loop.range,
                .form = "for",
                .invariants = loop.invariants,
                .condition = @as(?ast.ExprId, null),
                .for_stmt = @as(?ast.ForStmt, loop),
            };
        },
        .Switch => |loop| blk: {
            std.debug.assert(loop.label != null);
            for (loop.arms) |arm| try loop_bodies.append(arena, arm.body);
            if (loop.else_body) |else_body| try loop_bodies.append(arena, else_body);
            break :blk .{
                .range = loop.range,
                .form = "labeled_switch",
                .invariants = loop.invariants,
                .condition = @as(?ast.ExprId, loop.condition),
                .for_stmt = @as(?ast.ForStmt, null),
            };
        },
        else => unreachable,
    };
    const loc = sources.lineColumn(.{ .file_id = file_id, .range = loop_data.range });

    var facts = BodyFacts{
        .allocator = arena,
        .ast_file = ast_file,
        .typecheck = typecheck,
    };
    for (loop_bodies.items) |body| {
        try ast.walk.walkBody(BodyFacts, &facts, ast_file, body, inventory_walk_options);
    }
    const has_body_branch = if (findLoopCfgFact(loop_cfg_facts, raw.statement_id.index())) |cfg_fact|
        cfg_fact.has_body_branch
    else
        facts.has_body_branch;

    var loop_var_types: std.ArrayList([]const u8) = .empty;
    defer loop_var_types.deinit(arena);

    var invariant_support: std.ArrayList(InvariantSupport) = .empty;
    defer invariant_support.deinit(arena);
    var all_invariants_denotable = true;
    var first_unsupported_reason: ?[]const u8 = null;
    for (loop_data.invariants) |expr_id| {
        const expr_range = source.rangeOf(ast_file.expression(expr_id).*);
        const expr_loc = sources.lineColumn(.{ .file_id = file_id, .range = expr_range });
        const support = classifyInvariantSupport(
            formal_set,
            sources.file(file_id).path,
            raw.function_name,
            expr_loc.line,
            expr_loc.column,
        );
        const reason = if (support.reason) |value| try arena.dupe(u8, value) else null;
        if (!support.supported) {
            all_invariants_denotable = false;
            if (first_unsupported_reason == null) first_unsupported_reason = reason;
        }
        try invariant_support.append(arena, .{
            .line = expr_loc.line,
            .column = expr_loc.column,
            .supported = support.supported,
            .reason = reason,
        });
    }

    var effect_found = false;
    var effect: sema.Effect = .pure;
    for (typecheck.loop_body_effects) |entry| {
        if (entry.statement_id != raw.statement_id) continue;
        effect_found = true;
        effect = entry.effect;
        for (entry.variable_types) |ty| try loop_var_types.append(arena, try typeLabel(arena, ty));
        break;
    }
    const writes_storage = effectHasPersistentOrTransient(effect.writeSlots());
    const reads_storage = effectHasPersistentOrTransient(effect.readSlots());

    const formula_supported = if (loop_data.condition) |condition|
        if (std.mem.eql(u8, loop_data.form, "labeled_switch"))
            scalarTerm(ast_file, typecheck, condition)
        else
            scalarFormula(ast_file, typecheck, condition)
    else if (loop_data.for_stmt) |for_stmt|
        for_stmt.range_end != null and scalarFormula(ast_file, typecheck, for_stmt.iterable) and scalarFormula(ast_file, typecheck, for_stmt.range_end.?)
    else
        false;

    var shape_excluded: std.ArrayList([]const u8) = .empty;
    defer shape_excluded.deinit(arena);
    if (loop_data.invariants.len == 0) try appendReason(&shape_excluded, arena, "loop_missing_invariant");
    if (!formula_supported) try appendReason(&shape_excluded, arena, "loop_formula_unsupported:condition");
    if (std.mem.eql(u8, loop_data.form, "labeled_switch")) try appendReason(&shape_excluded, arena, "loop_form_unsupported:labeled_switch");
    if (!allTypesU256(loop_var_types.items)) try appendReason(&shape_excluded, arena, "loop_variable_not_u256");
    if (!facts.scalar_updates) try appendReason(&shape_excluded, arena, "loop_update_not_scalar_assignment");
    if (!effect_found) try appendReason(&shape_excluded, arena, "loop_effect_identity_missing");
    if (writes_storage) try appendReason(&shape_excluded, arena, "loop_has_storage_write");
    if (reads_storage) try appendReason(&shape_excluded, arena, "loop_has_storage_read");
    if (effect.hasExternal()) try appendReason(&shape_excluded, arena, "loop_has_external_call");
    if (facts.has_resource_operation) try appendReason(&shape_excluded, arena, "loop_has_resource_operation");
    if (facts.has_break_or_continue) try appendReason(&shape_excluded, arena, "loop_has_break_or_continue");
    if (facts.has_error_control_flow) try appendReason(&shape_excluded, arena, "loop_has_error_control_flow");
    if (raw.is_nested or facts.has_nested_loop) try appendReason(&shape_excluded, arena, "loop_has_nested_loop");
    if (has_body_branch) try appendReason(&shape_excluded, arena, "loop_has_branching_body");

    var excluded: std.ArrayList([]const u8) = .empty;
    defer excluded.deinit(arena);
    for (shape_excluded.items) |reason| try appendReason(&excluded, arena, reason);
    if (!all_invariants_denotable) {
        const reason = first_unsupported_reason orelse "unknown_reason";
        try appendReason(&excluded, arena, try std.fmt.allocPrint(arena, "loop_formula_unsupported:{s}", .{reason}));
    }

    return .{
        .statement_id = raw.statement_id,
        .line = loc.line,
        .column = loc.column,
        .function_name = raw.function_name,
        .loop_form = loop_data.form,
        .invariants = loop_data.invariants,
        .is_nested = raw.is_nested,
        .effect_found = effect_found,
        .writes_storage = writes_storage,
        .reads_storage = reads_storage,
        .has_calls = facts.has_calls,
        .has_body_branch = has_body_branch,
        .has_resource_operation = facts.has_resource_operation,
        .has_break_or_continue = facts.has_break_or_continue,
        .has_error_control_flow = facts.has_error_control_flow,
        .has_nested_loop = facts.has_nested_loop,
        .scalar_updates = facts.scalar_updates,
        .loop_var_types = try loop_var_types.toOwnedSlice(arena),
        .invariant_support = try invariant_support.toOwnedSlice(arena),
        .shape_scalar_fragment = shape_excluded.items.len == 0,
        .shape_excluded_by = try shape_excluded.toOwnedSlice(arena),
        .scalar_fragment = excluded.items.len == 0,
        .excluded_by = try excluded.toOwnedSlice(arena),
    };
}

fn collectLoopCfgFacts(allocator: std.mem.Allocator, module: mlir.MlirModule) ![]const LoopCfgFact {
    var facts: std.ArrayList(LoopCfgFact) = .empty;
    errdefer facts.deinit(allocator);
    try collectLoopCfgFactsInOperation(allocator, &facts, mlir.oraModuleGetOperation(module));
    std.mem.sort(LoopCfgFact, facts.items, {}, lessThanLoopCfgFact);
    return facts.toOwnedSlice(allocator);
}

fn collectLoopCfgFactsInOperation(
    allocator: std.mem.Allocator,
    facts: *std.ArrayList(LoopCfgFact),
    operation: mlir.MlirOperation,
) !void {
    const name_ref = mlir.oraOperationGetName(operation);
    defer mlir_c_api.freeStringRef(name_ref);
    const name = if (name_ref.data == null) "" else name_ref.data[0..name_ref.length];

    const body = if (std.mem.eql(u8, name, "scf.while"))
        mlir.oraScfWhileOpGetAfterBlock(operation)
    else if (std.mem.eql(u8, name, "scf.for"))
        mlir.oraScfForOpGetBodyBlock(operation)
    else
        null;
    if (body) |loop_body| {
        if (loopStatementIndex(operation)) |statement_index| {
            const has_body_branch = blockHasPathSplit(loop_body);
            if (findLoopCfgFactMutable(facts.items, statement_index)) |existing| {
                existing.has_body_branch = existing.has_body_branch or has_body_branch;
            } else {
                try facts.append(allocator, .{
                    .statement_index = statement_index,
                    .has_body_branch = has_body_branch,
                });
            }
        }
    }

    const region_count = mlir.oraOperationGetNumRegions(operation);
    for (0..region_count) |region_index| {
        var block = mlir.oraRegionGetFirstBlock(mlir.oraOperationGetRegion(operation, region_index));
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                try collectLoopCfgFactsInOperation(allocator, facts, child);
            }
        }
    }
}

fn blockHasPathSplit(block: mlir.MlirBlock) bool {
    var operation = mlir.oraBlockGetFirstOperation(block);
    while (!mlir.oraOperationIsNull(operation)) : (operation = mlir.oraOperationGetNextInBlock(operation)) {
        if (operationIntroducesPathSplit(operation)) return true;
        const region_count = mlir.oraOperationGetNumRegions(operation);
        for (0..region_count) |region_index| {
            var nested_block = mlir.oraRegionGetFirstBlock(mlir.oraOperationGetRegion(operation, region_index));
            while (!mlir.oraBlockIsNull(nested_block)) : (nested_block = mlir.oraBlockGetNextInRegion(nested_block)) {
                if (blockHasPathSplit(nested_block)) return true;
            }
        }
    }
    return false;
}

fn operationIntroducesPathSplit(operation: mlir.MlirOperation) bool {
    if (mlir.mlirOperationGetNumSuccessors(operation) > 1) return true;
    if (mlir.oraOperationGetNumRegions(operation) > 1) return true;

    const name_ref = mlir.oraOperationGetName(operation);
    defer mlir_c_api.freeStringRef(name_ref);
    if (name_ref.data == null) return false;
    const name = name_ref.data[0..name_ref.length];
    return std.mem.eql(u8, name, "scf.if") or
        std.mem.eql(u8, name, "cf.cond_br") or
        std.mem.eql(u8, name, "ora.switch") or
        std.mem.eql(u8, name, "ora.try_stmt") or
        std.mem.eql(u8, name, "ora.conditional_return");
}

fn loopStatementIndex(operation: mlir.MlirOperation) ?usize {
    const location = mlir.oraOperationGetLocation(operation);
    if (mlir.oraLocationIsNull(location)) return null;
    const location_ref = mlir.oraLocationPrintToString(location);
    defer mlir_c_api.freeStringRef(location_ref);
    if (location_ref.data == null or location_ref.length == 0) return null;

    const text = location_ref.data[0..location_ref.length];
    const marker = "ora.origin_stmt.";
    const start = std.mem.indexOf(u8, text, marker) orelse return null;
    const digits = text[start + marker.len ..];
    const end = std.mem.indexOfNone(u8, digits, "0123456789") orelse digits.len;
    if (end == 0) return null;
    return std.fmt.parseInt(usize, digits[0..end], 10) catch null;
}

fn findLoopCfgFact(facts: []const LoopCfgFact, statement_index: usize) ?LoopCfgFact {
    for (facts) |fact| if (fact.statement_index == statement_index) return fact;
    return null;
}

fn findLoopCfgFactMutable(facts: []LoopCfgFact, statement_index: usize) ?*LoopCfgFact {
    for (facts) |*fact| if (fact.statement_index == statement_index) return fact;
    return null;
}

fn lessThanLoopCfgFact(_: void, lhs: LoopCfgFact, rhs: LoopCfgFact) bool {
    return lhs.statement_index < rhs.statement_index;
}

const inventory_walk_options: ast.walk.WalkOptions = .{
    .enter_comptime_bodies = true,
    .enter_quantified_bodies = true,
};

const LoopInventoryVisitor = struct {
    allocator: std.mem.Allocator,
    loops: *std.ArrayList(RawLoop),
    seen: []bool,
    function_name: []const u8,
    loop_depth: u32 = 0,

    pub fn enterStmt(self: *@This(), ast_file: *const ast.AstFile, statement_id: ast.StmtId) !ast.walk.WalkControl {
        switch (ast_file.statement(statement_id).*) {
            .While, .For => {
                if (!self.seen[statement_id.index()]) {
                    self.seen[statement_id.index()] = true;
                    try self.loops.append(self.allocator, .{
                        .statement_id = statement_id,
                        .function_name = self.function_name,
                        .is_nested = self.loop_depth != 0,
                    });
                }
                self.loop_depth += 1;
            },
            .Switch => |switch_stmt| {
                if (switch_stmt.label == null) return .descend;
                if (!self.seen[statement_id.index()]) {
                    self.seen[statement_id.index()] = true;
                    try self.loops.append(self.allocator, .{
                        .statement_id = statement_id,
                        .function_name = self.function_name,
                        .is_nested = self.loop_depth != 0,
                    });
                }
                self.loop_depth += 1;
            },
            else => {},
        }
        return .descend;
    }

    pub fn exitStmt(self: *@This(), ast_file: *const ast.AstFile, statement_id: ast.StmtId) !void {
        switch (ast_file.statement(statement_id).*) {
            .While, .For => self.loop_depth -= 1,
            .Switch => |switch_stmt| if (switch_stmt.label != null) {
                self.loop_depth -= 1;
            },
            else => {},
        }
    }
};

const BodyFacts = struct {
    allocator: std.mem.Allocator,
    ast_file: *const ast.AstFile,
    typecheck: *const sema.TypeCheckResult,
    has_calls: bool = false,
    has_body_branch: bool = false,
    has_resource_operation: bool = false,
    has_break_or_continue: bool = false,
    has_error_control_flow: bool = false,
    has_nested_loop: bool = false,
    scalar_updates: bool = true,
    assigned_patterns: std.ArrayList(ast.PatternId) = .empty,

    pub fn enterStmt(self: *@This(), file: *const ast.AstFile, statement_id: ast.StmtId) !ast.walk.WalkControl {
        switch (file.statement(statement_id).*) {
            .VariableDecl => |decl| {
                if (decl.storage_class != .none and decl.storage_class != .memory) self.scalar_updates = false;
                if (file.pattern(decl.pattern).* != .Name) self.scalar_updates = false;
                if (!isU256(self.typecheck.pattern_types[decl.pattern.index()].type)) self.scalar_updates = false;
                if (decl.value) |value| {
                    if (!scalarTerm(file, self.typecheck, value)) self.scalar_updates = false;
                } else self.scalar_updates = false;
            },
            .Assign => |assign| {
                try appendPatternUnique(&self.assigned_patterns, self.allocator, assign.target);
                if (file.pattern(assign.target).* != .Name or !scalarTerm(file, self.typecheck, assign.value)) {
                    self.scalar_updates = false;
                }
            },
            .If => {
                self.has_body_branch = true;
                self.scalar_updates = false;
            },
            .Switch => |switch_stmt| {
                self.has_body_branch = true;
                if (switch_stmt.label != null) self.has_nested_loop = true;
                self.scalar_updates = false;
            },
            .Try => {
                self.has_body_branch = true;
                self.has_error_control_flow = true;
                self.scalar_updates = false;
            },
            .While, .For => {
                self.has_nested_loop = true;
                self.scalar_updates = false;
            },
            .Break, .Continue => {
                self.has_break_or_continue = true;
                self.scalar_updates = false;
            },
            .Return, .Log, .Lock, .Unlock, .CallHint, .Assert, .Assume, .Havoc, .Expr, .Error => self.scalar_updates = false,
            .Block, .LabeledBlock => {},
        }
        return .descend;
    }

    pub fn enterExpr(self: *@This(), file: *const ast.AstFile, expr_id: ast.ExprId) ast.walk.WalkControl {
        switch (file.expression(expr_id).*) {
            .Call => self.has_calls = true,
            .Switch => self.has_body_branch = true,
            .ErrorReturn => self.has_error_control_flow = true,
            .Unary => |unary| if (unary.op == .try_) {
                self.has_error_control_flow = true;
            },
            .Builtin => |builtin| if (ora_root.builtins.kindForName(builtin.name)) |kind| switch (kind) {
                .resource_amount, .resource_create, .resource_destroy, .resource_move => self.has_resource_operation = true,
                else => {},
            },
            else => {},
        }
        return .descend;
    }
};

fn classifyInvariantSupport(
    set_opt: ?obligation.ObligationSet,
    file_path: []const u8,
    function_name: []const u8,
    line: u32,
    column: u32,
) struct { supported: bool, reason: ?[]const u8 } {
    const set = set_opt orelse return .{ .supported = false, .reason = "formal_support_unavailable" };
    for (set.queries) |query| {
        if (query.logical_role != .invariant) continue;
        const query_file = query.source.file orelse continue;
        if (!pathsReferToSameSource(file_path, query_file)) continue;
        if (!queryOwnerMatchesFunction(query.owner, function_name)) continue;
        if (query.source.line != line or query.source.column != column) continue;
        var formula_only = query;
        formula_only.assumption_ids = &.{};
        return switch (obligation_to_lean.querySemanticSupport(set, formula_only)) {
            .supported => .{ .supported = true, .reason = null },
            .unsupported => |reason| .{ .supported = false, .reason = @tagName(std.meta.activeTag(reason)) },
        };
    }
    return .{ .supported = false, .reason = "formal_invariant_query_missing" };
}

fn queryOwnerMatchesFunction(owner: obligation.Owner, function_name: []const u8) bool {
    return switch (owner) {
        .function => |function| std.mem.eql(u8, function.name, function_name),
        .statement => |statement| std.mem.eql(u8, statement.function_name, function_name),
        else => false,
    };
}

fn scalarFormula(file: *const ast.AstFile, typecheck: *const sema.TypeCheckResult, expr_id: ast.ExprId) bool {
    if (typecheck.exprType(expr_id).kind() != .bool) return false;
    return scalarTerm(file, typecheck, expr_id);
}

fn scalarTerm(file: *const ast.AstFile, typecheck: *const sema.TypeCheckResult, expr_id: ast.ExprId) bool {
    const ty = typecheck.exprType(expr_id);
    if (ty.kind() != .bool and !isU256(ty) and ty.kind() != .comptime_integer) return false;
    return switch (file.expression(expr_id).*) {
        .IntegerLiteral, .BoolLiteral, .Name => true,
        .Unary => |unary| unary.op != .try_ and scalarTerm(file, typecheck, unary.operand),
        .Binary => |binary| scalarTerm(file, typecheck, binary.lhs) and scalarTerm(file, typecheck, binary.rhs),
        .Group => |group| scalarTerm(file, typecheck, group.expr),
        else => false,
    };
}

fn isU256(ty: sema.Type) bool {
    const base = if (ty.refinementBaseType()) |refinement_base| refinement_base.* else ty;
    return switch (base) {
        .integer => |integer| integer.isUnsignedBits(256),
        else => false,
    };
}

fn typeLabel(arena: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    if (ty.name()) |name| return try arena.dupe(u8, name);
    return switch (ty) {
        .integer => |integer| try std.fmt.allocPrint(arena, "{s}{d}", .{ if (integer.signed) "i" else "u", integer.bits }),
        else => try arena.dupe(u8, @tagName(ty.kind())),
    };
}

fn allTypesU256(types: []const []const u8) bool {
    for (types) |label| if (!std.mem.eql(u8, label, "u256")) return false;
    return true;
}

fn effectHasPersistentOrTransient(slots: []const sema.EffectSlot) bool {
    for (slots) |slot| if (slot.region == .storage or slot.region == .transient) return true;
    return false;
}

fn appendPatternUnique(list: *std.ArrayList(ast.PatternId), allocator: std.mem.Allocator, pattern: ast.PatternId) !void {
    for (list.items) |existing| if (existing == pattern) return;
    try list.append(allocator, pattern);
}

fn appendReason(list: *std.ArrayList([]const u8), allocator: std.mem.Allocator, reason: []const u8) !void {
    for (list.items) |existing| if (std.mem.eql(u8, existing, reason)) return;
    try list.append(allocator, reason);
}

fn findRecord(records: []LoopRecord, statement_index: u32) ?*LoopRecord {
    for (records) |*record| if (record.statement_id.index() == statement_index) return record;
    return null;
}

fn pathsReferToSameSource(lhs: []const u8, rhs: []const u8) bool {
    return std.mem.eql(u8, lhs, rhs);
}

fn lessThanRawLoop(_: void, lhs: RawLoop, rhs: RawLoop) bool {
    return lhs.statement_id.index() < rhs.statement_id.index();
}

fn writeLoopRecord(writer: anytype, file_path: []const u8, record: LoopRecord) !void {
    try writer.writeAll("{\"file\":");
    try writeJsonString(writer, file_path);
    try writer.print(",\"line\":{d},\"column\":{d},\"function\":", .{ record.line, record.column });
    try writeJsonString(writer, record.function_name);
    try writer.writeAll(",\"loop_form\":");
    try writeJsonString(writer, record.loop_form);
    try writer.print(",\"has_invariants\":{s},\"invariant_count\":{d}", .{ if (record.invariants.len != 0) "true" else "false", record.invariants.len });
    try writer.print(",\"queries\":{{\"LoopInvariantStep\":{d},\"LoopBodySafety\":{d},\"LoopInvariantPost\":{d}}}", .{
        record.queries.loop_invariant_step,
        record.queries.loop_body_safety,
        record.queries.loop_invariant_post,
    });
    try writer.print(",\"writes_storage\":{s},\"reads_storage\":{s},\"has_calls\":{s},\"has_body_branch\":{s},\"is_nested\":{s}", .{
        boolJson(record.writes_storage),
        boolJson(record.reads_storage),
        boolJson(record.has_calls),
        boolJson(record.has_body_branch),
        boolJson(record.is_nested),
    });
    try writer.writeAll(",\"loop_var_types\":[");
    for (record.loop_var_types, 0..) |label, index| {
        if (index != 0) try writer.writeByte(',');
        try writeJsonString(writer, label);
    }
    try writer.writeAll("],\"invariants_denotable\":{\"verdict\":");
    const all_supported = allInvariantSupport(record.invariant_support);
    try writeJsonString(writer, if (all_supported) "supported" else "unsupported");
    try writer.writeAll(",\"invariants\":[");
    for (record.invariant_support, 0..) |support, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{{\"line\":{d},\"column\":{d},\"supported\":{s},\"reason\":", .{ support.line, support.column, boolJson(support.supported) });
        if (support.reason) |reason| try writeJsonString(writer, reason) else try writer.writeAll("null");
        try writer.writeByte('}');
    }
    try writer.writeAll("]},\"shape_scalar_fragment\":");
    try writer.writeAll(boolJson(record.shape_scalar_fragment));
    try writer.writeAll(",\"shape_excluded_by\":[");
    for (record.shape_excluded_by, 0..) |reason, index| {
        if (index != 0) try writer.writeByte(',');
        try writeJsonString(writer, reason);
    }
    try writer.writeAll("],\"scalar_fragment\":");
    try writer.writeAll(boolJson(record.scalar_fragment));
    try writer.writeAll(",\"excluded_by\":[");
    for (record.excluded_by, 0..) |reason, index| {
        if (index != 0) try writer.writeByte(',');
        try writeJsonString(writer, reason);
    }
    try writer.writeAll("]}");
}

fn allInvariantSupport(items: []const InvariantSupport) bool {
    for (items) |item| if (!item.supported) return false;
    return true;
}

fn boolJson(value: bool) []const u8 {
    return if (value) "true" else "false";
}

fn writeFailureJson(writer: anytype, requested_path: []const u8, diagnostic: []const u8) !void {
    try writer.writeAll("{\"schema\":");
    try writeJsonString(writer, schema);
    try writer.print(",\"schema_version\":{d},\"source_file\":", .{schema_version});
    try writeJsonString(writer, requested_path);
    try writer.writeAll(",\"compiling\":false,\"first_diagnostic\":");
    try writeJsonString(writer, diagnostic);
    try writer.writeAll(",\"measurement_errors\":[],\"loops\":[]}\n");
}

fn firstCompilationDiagnostic(compilation: *compiler.driver.Compilation) !?[]const u8 {
    const package = compilation.db.sources.package(compilation.package_id);
    for (package.modules.items) |module_id| {
        const module = compilation.db.sources.module(module_id);
        if (firstErrorMessage(try compilation.db.syntaxDiagnostics(module.file_id))) |message| return message;
    }
    for (package.modules.items) |module_id| {
        const module = compilation.db.sources.module(module_id);
        if (firstErrorMessage(try compilation.db.astDiagnostics(module.file_id))) |message| return message;
    }
    for (package.modules.items) |module_id| {
        if (firstErrorMessage(try compilation.db.resolutionDiagnostics(module_id))) |message| return message;
    }
    for (package.modules.items) |module_id| {
        const result = try compilation.db.moduleTypeCheck(module_id);
        if (firstErrorMessage(&result.diagnostics)) |message| return message;
    }
    const lowering = try compilation.db.lowerToHir(compilation.root_module_id);
    return firstErrorMessage(&lowering.diagnostics);
}

fn firstErrorMessage(list: *const compiler.diagnostics.DiagnosticList) ?[]const u8 {
    for (list.items.items) |diagnostic| if (diagnostic.severity == .Error) return diagnostic.message;
    return null;
}

fn blockReasonName(reason: compiler.driver.ArtifactEmissionBlockReason) []const u8 {
    return switch (reason) {
        .package_diagnostics => "package_diagnostics",
        .hir_diagnostics => "hir_diagnostics",
        .hir_executable_fallbacks => "hir_executable_fallbacks",
        .structural_executable_fallback => "structural_executable_fallback",
    };
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| switch (byte) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0...8, 11...12, 14...0x1f => try writer.print("\\u00{x:0>2}", .{byte}),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

test "loop census schema is pinned" {
    try std.testing.expectEqualStrings("ora.loop_census.file.v2", schema);
    try std.testing.expectEqual(@as(u32, 2), schema_version);
}

test "loop census records the countThree fixture loop" {
    var compilation = try compiler.compilePackageWithOptions(
        std.testing.allocator,
        "ora-example/corpus/declarations/inline_functions.ora",
        .{ .compile_options = .{ .measure_loop_census = true } },
    );
    defer compilation.deinit();

    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();
    try writeCompilationJson(
        &output.writer,
        std.testing.allocator,
        "ora-example/corpus/declarations/inline_functions.ora",
        &compilation,
    );
    const json = try output.toOwnedSlice();
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"function\":\"countThree\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"line\":88") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"loop_form\":\"while\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"invariant_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"loop_var_types\":[\"u256\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"shape_scalar_fragment\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"scalar_fragment\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "loop_formula_unsupported:formal_invariant_query_missing") != null);
}

test "loop census attributes a post query to its source loop" {
    var compilation = try compiler.compilePackageWithOptions(
        std.testing.allocator,
        "ora-example/smt/verification/loop_invariants.ora",
        .{ .compile_options = .{ .measure_loop_census = true } },
    );
    defer compilation.deinit();

    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();
    try writeCompilationJson(
        &output.writer,
        std.testing.allocator,
        "ora-example/smt/verification/loop_invariants.ora",
        &compilation,
    );
    const json = try output.toOwnedSlice();
    defer std.testing.allocator.free(json);

    const record_start = std.mem.indexOf(u8, json, "\"line\":58") orelse return error.TestExpectedEqual;
    const record_tail = json[record_start..];
    const record_end = std.mem.indexOf(u8, record_tail, "},{\"file\"") orelse record_tail.len;
    try std.testing.expect(std.mem.indexOf(u8, record_tail[0..record_end], "\"LoopInvariantPost\":1") != null);
}
