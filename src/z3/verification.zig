//===----------------------------------------------------------------------===//
//
// Verification Pass Coordinator
//
//===----------------------------------------------------------------------===//
//
// Orchestrates the formal verification process:
// 1. Extract verification annotations from AST/MLIR
// 2. Encode to SMT
// 3. Query Z3
// 4. Report results
//
//===----------------------------------------------------------------------===//

const std = @import("std");
const z3 = @import("c.zig");
const mlir = @import("mlir_c_api").c;
const Context = @import("context.zig").Context;
const Solver = @import("solver.zig").Solver;
const Encoder = @import("encoder.zig").Encoder;
const errors = @import("errors.zig");
const mlir_helpers = @import("mlir_helpers.zig");
const ManagedArrayList = std.array_list.Managed;

pub const AnnotationKind = enum {
    Requires, // Function precondition
    Ensures, // Function postcondition
    Guard, // Runtime-enforced precondition / guard clause
    LoopInvariant, // Loop invariant
    ContractInvariant, // Contract-level invariant (future)
    RefinementGuard, // Runtime refinement guard
    Assume, // Verification-only assumption
    PathAssume, // Compiler-injected path-local assumption
};

const AssumptionKind = enum {
    requires,
    loop_invariant,
    path_assume,
    goal,
};

const AssumptionTag = struct {
    kind: AssumptionKind,
    function_name: []const u8,
    file: []const u8,
    line: u32,
    column: u32,
    label: []const u8,
    callee_name: ?[]const u8 = null,
    guard_id: ?[]const u8 = null,
    loop_owner: ?u64 = null,
};

const TrackedAssumption = struct {
    proxy: ?z3.Z3_ast = null,
    ast: z3.Z3_ast,
    tag: AssumptionTag,
};

fn cloneAssumptionTagSlice(
    allocator: std.mem.Allocator,
    tags: []const AssumptionTag,
) ![]const AssumptionTag {
    return if (tags.len == 0) &.{} else try allocator.dupe(AssumptionTag, tags);
}

pub const SmtReportArtifacts = struct {
    markdown: []u8,
    json: []u8,

    pub fn deinit(self: *SmtReportArtifacts, allocator: std.mem.Allocator) void {
        allocator.free(self.markdown);
        allocator.free(self.json);
    }
};

/// Verification pass for MLIR modules
pub const VerificationPass = struct {
    const SwitchCaseMetadata = struct {
        case_kinds: []i64,
        case_values: []i64,
        range_starts: []i64,
        range_ends: []i64,
        default_case_index: ?i64,

        fn deinit(self: *SwitchCaseMetadata, allocator: std.mem.Allocator) void {
            allocator.free(self.case_kinds);
            allocator.free(self.case_values);
            allocator.free(self.range_starts);
            allocator.free(self.range_ends);
        }
    };

    const SavedValueBinding = struct {
        value_id: u64,
        had_binding: bool,
        old_binding: z3.Z3_ast,
    };

    pub const VerifyMode = enum {
        Basic,
        Full,
    };

    context: *Context,
    solver: Solver,
    encoder: Encoder,
    allocator: std.mem.Allocator,
    debug_z3: bool,
    trace_smt: bool,
    trace_smtlib: bool,
    phase_debug: bool,
    random_seed: u32,
    timeout_ms: ?u32,
    parallel: bool,
    max_workers: usize,
    filter_function_name: ?[]const u8 = null,
    verify_mode: VerifyMode = .Full,
    verify_calls: bool = true,
    verify_state: bool = true,
    verify_stats: bool = false,
    explain_cores: bool = false,

    /// Current function being processed (for MLIR extraction)
    current_function_name: ?[]const u8 = null,
    /// Whether the current function should produce top-level verification queries.
    current_function_verify_enabled: bool = true,

    /// Encoded annotations collected from MLIR
    encoded_annotations: ManagedArrayList(EncodedAnnotation),
    /// Path assumptions active in the current MLIR traversal context.
    active_path_assumptions: ManagedArrayList(ActivePathAssume),

    /// Storage for duplicated function names
    function_name_storage: ManagedArrayList([]const u8),
    /// Storage for duplicated location file names
    location_storage: ManagedArrayList([]const u8),
    /// Storage for duplicated guard ids
    guard_id_storage: ManagedArrayList([]const u8),
    /// Storage for duplicated annotation labels/messages
    label_storage: ManagedArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator) !VerificationPass {
        var context = try allocator.create(Context);
        errdefer allocator.destroy(context);
        context.* = try Context.init(allocator);
        errdefer context.deinit();

        var solver = try Solver.init(context, allocator);
        errdefer solver.deinit();

        var encoder = Encoder.init(context, allocator);
        const debug_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_DEBUG") catch null;
        const debug_z3 = debug_env != null;
        if (debug_env) |val| allocator.free(val);

        const trace_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_TRACE") catch null;
        const trace_smt = if (trace_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else false;

        const trace_smtlib_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_TRACE_SMTLIB") catch null;
        const trace_smtlib = if (trace_smtlib_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else false;

        const phase_debug_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_PHASE_DEBUG") catch null;
        const phase_debug = if (phase_debug_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else false;

        const timeout_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_TIMEOUT_MS") catch null;
        var timeout_ms: ?u32 = 60_000;
        if (timeout_env) |val| {
            timeout_ms = std.fmt.parseInt(u32, val, 10) catch null;
            allocator.free(val);
        }

        const seed_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_SEED") catch null;
        var random_seed: u32 = 0;
        if (seed_env) |val| {
            random_seed = std.fmt.parseInt(u32, val, 10) catch random_seed;
            allocator.free(val);
        }

        const parallel_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_PARALLEL") catch null;
        const parallel = if (parallel_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else false;

        const workers_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_WORKERS") catch null;
        var max_workers: usize = std.Thread.getCpuCount() catch 1;
        if (workers_env) |val| {
            max_workers = std.fmt.parseInt(usize, val, 10) catch max_workers;
            allocator.free(val);
        }

        const verify_mode_env = std.process.getEnvVarOwned(allocator, "ORA_VERIFY_MODE") catch null;
        const verify_mode = if (verify_mode_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseVerifyMode(val);
        } else .Full;

        const verify_calls_env = std.process.getEnvVarOwned(allocator, "ORA_VERIFY_CALLS") catch null;
        const verify_calls = if (verify_calls_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else true;

        const verify_state_env = std.process.getEnvVarOwned(allocator, "ORA_VERIFY_STATE") catch null;
        const verify_state = if (verify_state_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else true;

        const verify_stats_env = std.process.getEnvVarOwned(allocator, "ORA_VERIFY_STATS") catch null;
        const verify_stats = if (verify_stats_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else false;

        const explain_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_EXPLAIN") catch null;
        const explain_cores = if (explain_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else false;

        encoder.setVerifyCalls(verify_calls);
        encoder.setVerifyState(verify_state);
        try solver.setRandomSeed(random_seed);

        if (timeout_ms) |ms| {
            try solver.setTimeoutMs(ms);
        }

        const pass = VerificationPass{
            .context = context,
            .solver = solver,
            .encoder = encoder,
            .allocator = allocator,
            .debug_z3 = debug_z3,
            .trace_smt = trace_smt,
            .trace_smtlib = trace_smtlib,
            .phase_debug = phase_debug,
            .random_seed = random_seed,
            .timeout_ms = timeout_ms,
            .parallel = parallel,
            .max_workers = max_workers,
            .verify_mode = verify_mode,
            .verify_calls = verify_calls,
            .verify_state = verify_state,
            .verify_stats = verify_stats,
            .explain_cores = explain_cores,
            .encoded_annotations = ManagedArrayList(EncodedAnnotation).init(allocator),
            .active_path_assumptions = ManagedArrayList(ActivePathAssume).init(allocator),
            .function_name_storage = ManagedArrayList([]const u8).init(allocator),
            .location_storage = ManagedArrayList([]const u8).init(allocator),
            .guard_id_storage = ManagedArrayList([]const u8).init(allocator),
            .label_storage = ManagedArrayList([]const u8).init(allocator),
        };
        return pass;
    }

    pub fn setVerifyMode(self: *VerificationPass, mode: VerifyMode) void {
        self.verify_mode = mode;
    }

    pub fn setVerifyCalls(self: *VerificationPass, enabled: bool) void {
        self.verify_calls = enabled;
        self.encoder.setVerifyCalls(enabled);
    }

    pub fn setVerifyState(self: *VerificationPass, enabled: bool) void {
        self.verify_state = enabled;
        self.encoder.setVerifyState(enabled);
    }

    pub fn setVerifyStats(self: *VerificationPass, enabled: bool) void {
        self.verify_stats = enabled;
    }

    pub fn setExplainCores(self: *VerificationPass, enabled: bool) void {
        self.explain_cores = enabled;
    }

    pub fn deinit(self: *VerificationPass) void {
        for (self.function_name_storage.items) |name| {
            self.allocator.free(name);
        }
        self.function_name_storage.deinit();
        for (self.location_storage.items) |name| {
            self.allocator.free(name);
        }
        self.location_storage.deinit();
        for (self.guard_id_storage.items) |name| {
            self.allocator.free(name);
        }
        self.guard_id_storage.deinit();
        for (self.label_storage.items) |label| {
            self.allocator.free(label);
        }
        self.label_storage.deinit();
        for (self.encoded_annotations.items) |ann| {
            if (ann.extra_constraints.len > 0) {
                self.allocator.free(ann.extra_constraints);
            }
            if (ann.old_extra_constraints.len > 0) {
                self.allocator.free(ann.old_extra_constraints);
            }
            if (ann.loop_entry_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_entry_extra_constraints);
            }
            if (ann.loop_step_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_step_extra_constraints);
            }
            if (ann.loop_step_head_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_step_head_extra_constraints);
            }
            if (ann.loop_step_body_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_step_body_extra_constraints);
            }
            if (ann.loop_exit_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_exit_extra_constraints);
            }
            if (ann.path_constraints.len > 0) {
                self.allocator.free(ann.path_constraints);
            }
        }
        self.encoded_annotations.deinit();
        self.releaseActivePathAssumptionsFrom(0);
        self.active_path_assumptions.deinit();
        self.encoder.deinit();
        self.solver.deinit();
        self.context.deinit();
        self.allocator.destroy(self.context);
    }

    fn releaseActivePathAssumptionsFrom(self: *VerificationPass, start_len: usize) void {
        if (start_len >= self.active_path_assumptions.items.len) {
            self.active_path_assumptions.shrinkRetainingCapacity(start_len);
            return;
        }
        for (self.active_path_assumptions.items[start_len..]) |assume| {
            if (assume.owned_extra_constraints and assume.extra_constraints.len > 0) {
                self.allocator.free(assume.extra_constraints);
            }
        }
        self.active_path_assumptions.shrinkRetainingCapacity(start_len);
    }

    fn cloneActivePathAssumeSlice(
        self: *VerificationPass,
        assumes: []const ActivePathAssume,
    ) ![]ActivePathAssume {
        if (assumes.len == 0) {
            return try self.allocator.alloc(ActivePathAssume, 0);
        }
        var dup = try self.allocator.alloc(ActivePathAssume, assumes.len);
        for (dup) |*entry| {
            entry.* = .{
                .condition = self.encoder.encodeBoolConstant(true),
                .extra_constraints = &[_]z3.Z3_ast{},
                .owned_extra_constraints = false,
            };
        }
        errdefer self.freeOwnedActivePathAssumeSlice(dup);
        for (assumes, 0..) |assume, idx| {
            dup[idx] = .{
                .condition = assume.condition,
                .extra_constraints = if (assume.owned_extra_constraints)
                    try self.cloneConstraintSlice(assume.extra_constraints)
                else
                    assume.extra_constraints,
                .owned_extra_constraints = assume.owned_extra_constraints,
            };
        }
        return dup;
    }

    fn freeOwnedActivePathAssumeSlice(self: *VerificationPass, assumes: []ActivePathAssume) void {
        for (assumes) |assume| {
            if (assume.owned_extra_constraints and assume.extra_constraints.len > 0) {
                self.allocator.free(assume.extra_constraints);
            }
        }
        self.allocator.free(assumes);
    }

    //===----------------------------------------------------------------------===//
    // annotation Extraction from MLIR
    //===----------------------------------------------------------------------===//

    /// Extract verification annotations from MLIR module
    /// This walks MLIR operations looking for ora.requires, ora.ensures, ora.invariant
    pub fn extractAnnotationsFromMLIR(self: *VerificationPass, mlir_module: mlir.MlirModule) !void {
        // get the module operation
        const module_op = mlir.oraModuleGetOperation(mlir_module);

        // Register all functions first so call summaries can resolve callees
        // regardless of definition order.
        try self.registerFunctionOps(module_op);

        // walk all regions in the module
        const num_regions = mlir.oraOperationGetNumRegions(module_op);
        for (0..@intCast(num_regions)) |region_idx| {
            const region = mlir.oraOperationGetRegion(module_op, @intCast(region_idx));
            try self.walkMLIRRegion(region);
        }
    }

    fn registerFunctionOps(self: *VerificationPass, root: mlir.MlirOperation) !void {
        const op_name_ref = self.getMLIROperationName(root);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];
        if (std.mem.eql(u8, op_name, "func.func")) {
            try self.encoder.registerFunctionOperation(root);
        } else if (std.mem.eql(u8, op_name, "ora.struct.decl")) {
            try self.encoder.registerStructDeclOperation(root);
        }

        const num_regions = mlir.oraOperationGetNumRegions(root);
        for (0..@intCast(num_regions)) |region_idx| {
            const region = mlir.oraOperationGetRegion(root, @intCast(region_idx));
            if (mlir.oraRegionIsNull(region)) continue;
            var block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(block)) {
                var op = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(op)) {
                    try self.registerFunctionOps(op);
                    op = mlir.oraOperationGetNextInBlock(op);
                }
                block = mlir.oraBlockGetNextInRegion(block);
            }
        }
    }

    /// Walk an MLIR region to find verification operations
    fn walkMLIRRegion(self: *VerificationPass, region: mlir.MlirRegion) !void {
        const region_path_assumption_len = self.active_path_assumptions.items.len;
        defer self.releaseActivePathAssumptionsFrom(region_path_assumption_len);

        var predecessor_counts = std.AutoHashMap(usize, usize).init(self.allocator);
        defer predecessor_counts.deinit();
        try self.collectRegionPredecessorCounts(region, &predecessor_counts);

        var inherited_assumptions = std.AutoHashMap(usize, []ActivePathAssume).init(self.allocator);
        defer {
            var it = inherited_assumptions.iterator();
            while (it.next()) |entry| {
                self.freeOwnedActivePathAssumeSlice(entry.value_ptr.*);
            }
            inherited_assumptions.deinit();
        }

        // get first block in region
        var current_block = mlir.oraRegionGetFirstBlock(region);

        while (!mlir.oraBlockIsNull(current_block)) {
            self.releaseActivePathAssumptionsFrom(region_path_assumption_len);
            const block_key = @intFromPtr(current_block.ptr);
            if (inherited_assumptions.fetchRemove(block_key)) |entry| {
                defer self.allocator.free(entry.value);
                for (entry.value) |assume| {
                    try self.active_path_assumptions.append(assume);
                }
            }

            var block_arg_bindings = std.ArrayList(SavedValueBinding){};
            defer {
                for (block_arg_bindings.items) |saved| {
                    if (saved.had_binding) {
                        self.encoder.value_bindings.put(saved.value_id, saved.old_binding) catch {};
                    } else {
                        _ = self.encoder.value_bindings.remove(saved.value_id);
                    }
                }
                block_arg_bindings.deinit(self.allocator);
            }
            try self.bindRegionBlockArguments(current_block, &block_arg_bindings);

            var block_loop_invariant_annotations = std.ArrayList(usize){};
            defer block_loop_invariant_annotations.deinit(self.allocator);

            // walk operations in this block
            var current_op = mlir.oraBlockGetFirstOperation(current_block);

            while (!mlir.oraOperationIsNull(current_op)) {
                const op_name_ref = self.getMLIROperationName(current_op);
                defer @import("mlir_c_api").freeStringRef(op_name_ref);
                const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
                    ""
                else
                    op_name_ref.data[0..op_name_ref.length];
                const prev_function = self.current_function_name;
                const prev_verify_enabled = self.current_function_verify_enabled;
                if (std.mem.eql(u8, op_name, "func.func")) {
                    if (try self.getFunctionNameFromOp(current_op)) |fn_name| {
                        self.current_function_name = fn_name;
                        self.current_function_verify_enabled = self.shouldVerifyFunctionOp(current_op);
                        if (self.filter_function_name) |target_fn| {
                            if (std.mem.eql(u8, fn_name, target_fn)) {
                                self.current_function_verify_enabled = true;
                            }
                        }
                        self.encoder.resetFunctionState();
                    }
                }

                if (try self.processMLIROperation(current_op)) |annotation_index| {
                    if (self.encoded_annotations.items[annotation_index].kind == .LoopInvariant) {
                        try block_loop_invariant_annotations.append(self.allocator, annotation_index);
                    }
                }

                if (std.mem.eql(u8, op_name, "func.func") and !self.current_function_verify_enabled) {
                    // Skip traversing function bodies that are not externally reachable
                    // entrypoints (e.g. private/internal helpers).
                } else if (std.mem.eql(u8, op_name, "scf.if")) {
                    try self.walkScfIfRegions(current_op);
                } else if (std.mem.eql(u8, op_name, "ora.conditional_return")) {
                    try self.walkConditionalReturnRegions(current_op);
                } else if (std.mem.eql(u8, op_name, "ora.switch")) {
                    try self.walkOraSwitchRegions(current_op);
                } else {
                    // walk nested regions (for functions, loops, etc.)
                    const num_regions = mlir.oraOperationGetNumRegions(current_op);
                    for (0..@intCast(num_regions)) |region_idx| {
                        const nested_region = mlir.oraOperationGetRegion(current_op, @intCast(region_idx));
                        try self.walkMLIRRegion(nested_region);
                    }
                }

                if (std.mem.eql(u8, op_name, "func.func")) {
                    self.current_function_name = prev_function;
                    self.current_function_verify_enabled = prev_verify_enabled;
                }

                current_op = mlir.oraOperationGetNextInBlock(current_op);
            }

            if (block_loop_invariant_annotations.items.len > 0) {
                try self.refreshLoopInvariantStepConditions(block_loop_invariant_annotations.items);
            }

            // Propagate accumulated path assumptions to CFG successors with a
            // single predecessor. This keeps linear guard chains precise while
            // staying conservative on merge blocks.
            const term = mlir.oraBlockGetTerminator(current_block);
            if (!mlir.oraOperationIsNull(term)) {
                const num_succ = mlir.mlirOperationGetNumSuccessors(term);
                if (num_succ > 0) {
                    const block_assumptions = self.active_path_assumptions.items[region_path_assumption_len..];
                    for (0..@intCast(num_succ)) |succ_idx| {
                        const succ_block = mlir.mlirOperationGetSuccessor(term, @intCast(succ_idx));
                        if (mlir.oraBlockIsNull(succ_block)) continue;
                        const succ_key = @intFromPtr(succ_block.ptr);
                        const pred_count = predecessor_counts.get(succ_key) orelse 0;
                        if (pred_count != 1) continue;

                        if (inherited_assumptions.fetchRemove(succ_key)) |removed| {
                            self.freeOwnedActivePathAssumeSlice(removed.value);
                        }
                        const dup = try self.cloneActivePathAssumeSlice(block_assumptions);
                        try inherited_assumptions.put(succ_key, dup);
                    }
                }
            }

            current_block = mlir.oraBlockGetNextInRegion(current_block);
        }
    }

    fn bindRegionBlockArguments(
        self: *VerificationPass,
        block: mlir.MlirBlock,
        saved_bindings: *std.ArrayList(SavedValueBinding),
    ) !void {
        const parent_op = mlir.mlirBlockGetParentOperation(block);
        if (mlir.oraOperationIsNull(parent_op)) return;

        const parent_name_ref = mlir.oraOperationGetName(parent_op);
        defer @import("mlir_c_api").freeStringRef(parent_name_ref);
        const parent_name = if (parent_name_ref.data == null or parent_name_ref.length == 0)
            ""
        else
            parent_name_ref.data[0..parent_name_ref.length];

        if (!std.mem.eql(u8, parent_name, "scf.while")) return;

        const after_block = mlir.oraScfWhileOpGetAfterBlock(parent_op);
        if (mlir.oraBlockIsNull(after_block) or after_block.ptr != block.ptr) return;
        const before_block = mlir.oraScfWhileOpGetBeforeBlock(parent_op);
        if (mlir.oraBlockIsNull(before_block)) return;

        const bind_count = @min(
            @as(usize, @intCast(mlir.oraBlockGetNumArguments(before_block))),
            @as(usize, @intCast(mlir.oraBlockGetNumArguments(after_block))),
        );

        var i: usize = 0;
        while (i < bind_count) : (i += 1) {
            const before_arg = mlir.oraBlockGetArgument(before_block, @intCast(i));
            const after_arg = mlir.oraBlockGetArgument(after_block, @intCast(i));
            const after_value_id = @intFromPtr(after_arg.ptr);
            if (self.encoder.value_bindings.get(after_value_id)) |old_binding| {
                try saved_bindings.append(self.allocator, .{
                    .value_id = after_value_id,
                    .had_binding = true,
                    .old_binding = old_binding,
                });
            } else {
                try saved_bindings.append(self.allocator, .{
                    .value_id = after_value_id,
                    .had_binding = false,
                    .old_binding = undefined,
                });
            }
            try self.encoder.bindValue(after_arg, try self.encoder.encodeValue(before_arg));
        }
    }

    const EncoderBranchState = struct {
        global_map: std.StringHashMap(z3.Z3_ast),
        global_old_map: std.StringHashMap(z3.Z3_ast),
        global_entry_map: std.StringHashMap(z3.Z3_ast),
        memref_map: std.AutoHashMap(u64, Encoder.TrackedMemrefState),
        value_map: std.AutoHashMap(u64, z3.Z3_ast),
        value_map_old: std.AutoHashMap(u64, z3.Z3_ast),
        value_bindings: std.AutoHashMap(u64, z3.Z3_ast),
        written_global_slots: std.StringHashMap(void),
        materialized_calls: std.AutoHashMap(u64, void),

        fn init(allocator: std.mem.Allocator) EncoderBranchState {
            return .{
                .global_map = std.StringHashMap(z3.Z3_ast).init(allocator),
                .global_old_map = std.StringHashMap(z3.Z3_ast).init(allocator),
                .global_entry_map = std.StringHashMap(z3.Z3_ast).init(allocator),
                .memref_map = std.AutoHashMap(u64, Encoder.TrackedMemrefState).init(allocator),
                .value_map = std.AutoHashMap(u64, z3.Z3_ast).init(allocator),
                .value_map_old = std.AutoHashMap(u64, z3.Z3_ast).init(allocator),
                .value_bindings = std.AutoHashMap(u64, z3.Z3_ast).init(allocator),
                .written_global_slots = std.StringHashMap(void).init(allocator),
                .materialized_calls = std.AutoHashMap(u64, void).init(allocator),
            };
        }

        fn deinit(self: *EncoderBranchState, allocator: std.mem.Allocator) void {
            var g_it = self.global_map.iterator();
            while (g_it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            self.global_map.deinit();

            var old_it = self.global_old_map.iterator();
            while (old_it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            self.global_old_map.deinit();

            var entry_it = self.global_entry_map.iterator();
            while (entry_it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            self.global_entry_map.deinit();

            self.memref_map.deinit();
            self.value_map.deinit();
            self.value_map_old.deinit();
            self.value_bindings.deinit();

            var written_it = self.written_global_slots.iterator();
            while (written_it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            self.written_global_slots.deinit();
            self.materialized_calls.deinit();
        }
    };

    fn captureEncoderBranchState(self: *VerificationPass) !EncoderBranchState {
        var snap = EncoderBranchState.init(self.allocator);
        errdefer snap.deinit(self.allocator);

        var g_it = self.encoder.global_map.iterator();
        while (g_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try snap.global_map.put(key_dup, entry.value_ptr.*);
        }

        var old_it = self.encoder.global_old_map.iterator();
        while (old_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try snap.global_old_map.put(key_dup, entry.value_ptr.*);
        }

        var entry_it = self.encoder.global_entry_map.iterator();
        while (entry_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try snap.global_entry_map.put(key_dup, entry.value_ptr.*);
        }

        var m_it = self.encoder.memref_map.iterator();
        while (m_it.next()) |entry| {
            try snap.memref_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var v_it = self.encoder.value_map.iterator();
        while (v_it.next()) |entry| {
            try snap.value_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var v_old_it = self.encoder.value_map_old.iterator();
        while (v_old_it.next()) |entry| {
            try snap.value_map_old.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var bind_it = self.encoder.value_bindings.iterator();
        while (bind_it.next()) |entry| {
            try snap.value_bindings.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var written_it = self.encoder.written_global_slots.iterator();
        while (written_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try snap.written_global_slots.put(key_dup, {});
        }

        var call_it = self.encoder.materialized_calls.iterator();
        while (call_it.next()) |entry| {
            try snap.materialized_calls.put(entry.key_ptr.*, {});
        }

        return snap;
    }

    fn clearEncoderGlobalMap(self: *VerificationPass) void {
        var g_it = self.encoder.global_map.iterator();
        while (g_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.encoder.global_map.clearRetainingCapacity();
    }

    fn restoreEncoderBranchState(self: *VerificationPass, snap: *const EncoderBranchState) !void {
        self.clearEncoderGlobalMap();

        var old_it = self.encoder.global_old_map.iterator();
        while (old_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.encoder.global_old_map.clearRetainingCapacity();

        var entry_it = self.encoder.global_entry_map.iterator();
        while (entry_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.encoder.global_entry_map.clearRetainingCapacity();

        self.encoder.memref_map.clearRetainingCapacity();

        self.encoder.value_map.clearRetainingCapacity();
        self.encoder.value_map_old.clearRetainingCapacity();
        self.encoder.value_bindings.clearRetainingCapacity();

        var written_it = self.encoder.written_global_slots.iterator();
        while (written_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.encoder.written_global_slots.clearRetainingCapacity();

        self.encoder.materialized_calls.clearRetainingCapacity();
        self.encoder.invalidateValueCaches();

        var g_it = snap.global_map.iterator();
        while (g_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try self.encoder.global_map.put(key_dup, entry.value_ptr.*);
        }

        var snap_old_it = snap.global_old_map.iterator();
        while (snap_old_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try self.encoder.global_old_map.put(key_dup, entry.value_ptr.*);
        }

        var snap_entry_it = snap.global_entry_map.iterator();
        while (snap_entry_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try self.encoder.global_entry_map.put(key_dup, entry.value_ptr.*);
        }

        var m_it = snap.memref_map.iterator();
        while (m_it.next()) |entry| {
            try self.encoder.memref_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var v_it = snap.value_map.iterator();
        while (v_it.next()) |entry| {
            try self.encoder.value_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var v_old_it = snap.value_map_old.iterator();
        while (v_old_it.next()) |entry| {
            try self.encoder.value_map_old.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var bind_it = snap.value_bindings.iterator();
        while (bind_it.next()) |entry| {
            try self.encoder.value_bindings.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var snap_written_it = snap.written_global_slots.iterator();
        while (snap_written_it.next()) |entry| {
            const key_dup = try self.allocator.dupe(u8, entry.key_ptr.*);
            try self.encoder.written_global_slots.put(key_dup, {});
        }

        var snap_call_it = snap.materialized_calls.iterator();
        while (snap_call_it.next()) |entry| {
            try self.encoder.materialized_calls.put(entry.key_ptr.*, {});
        }
    }

    fn mergeStableCurrentValueCachesIntoSnapshot(
        self: *VerificationPass,
        snap: *EncoderBranchState,
    ) !void {
        var value_it = self.encoder.value_map.iterator();
        while (value_it.next()) |entry| {
            const key = entry.key_ptr.*;
            if (snap.value_map.contains(key)) continue;
            const value = mlir.MlirValue{ .ptr = @ptrFromInt(key) };
            if (!self.encoder.valueIsStableAcrossAnnotationRestore(value)) continue;
            try snap.value_map.put(key, entry.value_ptr.*);
        }
    }

    fn appendUniqueName(self: *VerificationPass, list: *std.ArrayList([]const u8), name: []const u8) !void {
        for (list.items) |existing| {
            if (std.mem.eql(u8, existing, name)) return;
        }
        try list.append(self.allocator, name);
    }

    fn mergeEncoderBranchState(
        self: *VerificationPass,
        condition: z3.Z3_ast,
        base: *const EncoderBranchState,
        then_state: *const EncoderBranchState,
        else_state: *const EncoderBranchState,
    ) !void {
        // Globals: merge per-slot as ite(condition, then, else), defaulting to base.
        self.clearEncoderGlobalMap();
        var global_names = std.ArrayList([]const u8){};
        defer global_names.deinit(self.allocator);

        var base_g_it = base.global_map.iterator();
        while (base_g_it.next()) |entry| {
            try appendUniqueName(self, &global_names, entry.key_ptr.*);
        }
        var then_g_it = then_state.global_map.iterator();
        while (then_g_it.next()) |entry| {
            try appendUniqueName(self, &global_names, entry.key_ptr.*);
        }
        var else_g_it = else_state.global_map.iterator();
        while (else_g_it.next()) |entry| {
            try appendUniqueName(self, &global_names, entry.key_ptr.*);
        }

        for (global_names.items) |name| {
            const base_opt = base.global_map.get(name);
            const then_opt = then_state.global_map.get(name);
            const else_opt = else_state.global_map.get(name);
            const fallback = base_opt orelse blk: {
                const branch_val = then_opt orelse else_opt orelse break :blk null;
                const branch_sort = z3.Z3_get_sort(self.encoder.context.ctx, branch_val);
                break :blk try self.encoder.getOrCreateCurrentGlobal(name, branch_sort);
            } orelse continue;
            const then_val = then_opt orelse fallback;
            const else_val = else_opt orelse fallback;
            const merged = if (then_val == else_val)
                then_val
            else
                self.encoder.encodeIte(condition, then_val, else_val);

            if (self.encoder.global_map.getPtr(name)) |existing| {
                existing.* = merged;
            } else {
                const key_dup = try self.allocator.dupe(u8, name);
                try self.encoder.global_map.put(key_dup, merged);
            }
        }

        // Scalar memrefs: same merge logic.
        self.encoder.memref_map.clearRetainingCapacity();
        var mem_keys = std.AutoHashMap(u64, void).init(self.allocator);
        defer mem_keys.deinit();

        var base_m_it = base.memref_map.iterator();
        while (base_m_it.next()) |entry| {
            try mem_keys.put(entry.key_ptr.*, {});
        }
        var then_m_it = then_state.memref_map.iterator();
        while (then_m_it.next()) |entry| {
            try mem_keys.put(entry.key_ptr.*, {});
        }
        var else_m_it = else_state.memref_map.iterator();
        while (else_m_it.next()) |entry| {
            try mem_keys.put(entry.key_ptr.*, {});
        }

        var key_it = mem_keys.iterator();
        while (key_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const base_opt = base.memref_map.get(key);
            const then_opt = then_state.memref_map.get(key);
            const else_opt = else_state.memref_map.get(key);
            const fallback_value = if (base_opt) |base_state|
                base_state.value
            else if (then_opt) |then_state_entry|
                then_state_entry.value
            else if (else_opt) |else_state_entry|
                else_state_entry.value
            else
                continue;
            const fallback_init = if (base_opt) |base_state|
                base_state.initialized
            else
                z3.Z3_mk_false(self.context.ctx);
            const then_state_entry = then_opt orelse Encoder.TrackedMemrefState{
                .value = fallback_value,
                .initialized = fallback_init,
            };
            const else_state_entry = else_opt orelse Encoder.TrackedMemrefState{
                .value = fallback_value,
                .initialized = fallback_init,
            };
            const then_val = then_state_entry.value;
            const else_val = else_state_entry.value;
            const merged = if (then_val == else_val)
                then_val
            else
                self.encoder.encodeIte(condition, then_val, else_val);
            try self.encoder.memref_map.put(key, .{
                .value = merged,
                .initialized = self.encoder.mergeInitPredicate(condition, then_state_entry.initialized, else_state_entry.initialized),
            });
        }
    }

    fn mergeEncoderBranchStatesMany(
        self: *VerificationPass,
        conditions: []const z3.Z3_ast,
        base: *const EncoderBranchState,
        branch_states: []const EncoderBranchState,
    ) !void {
        self.clearEncoderGlobalMap();
        var global_names = std.ArrayList([]const u8){};
        defer global_names.deinit(self.allocator);

        var base_g_it = base.global_map.iterator();
        while (base_g_it.next()) |entry| {
            try self.appendUniqueName(&global_names, entry.key_ptr.*);
        }
        for (branch_states) |*branch_state| {
            var branch_it = branch_state.global_map.iterator();
            while (branch_it.next()) |entry| {
                try self.appendUniqueName(&global_names, entry.key_ptr.*);
            }
        }

        for (global_names.items) |name| {
            const base_opt = base.global_map.get(name);
            var fallback = base_opt orelse blk: {
                for (branch_states) |*branch_state| {
                    if (branch_state.global_map.get(name)) |value| {
                        const branch_sort = z3.Z3_get_sort(self.encoder.context.ctx, value);
                        break :blk try self.encoder.getOrCreateCurrentGlobal(name, branch_sort);
                    }
                }
                continue;
            };

            var idx = branch_states.len;
            while (idx > 0) {
                idx -= 1;
                const branch_val = branch_states[idx].global_map.get(name) orelse fallback;
                fallback = if (branch_val == fallback)
                    branch_val
                else
                    self.encoder.encodeIte(conditions[idx], branch_val, fallback);
            }

            if (self.encoder.global_map.getPtr(name)) |existing| {
                existing.* = fallback;
            } else {
                const key_dup = try self.allocator.dupe(u8, name);
                try self.encoder.global_map.put(key_dup, fallback);
            }
        }

        self.encoder.memref_map.clearRetainingCapacity();
        var mem_keys = std.AutoHashMap(u64, void).init(self.allocator);
        defer mem_keys.deinit();

        var base_m_it = base.memref_map.iterator();
        while (base_m_it.next()) |entry| {
            try mem_keys.put(entry.key_ptr.*, {});
        }
        for (branch_states) |*branch_state| {
            var branch_it = branch_state.memref_map.iterator();
            while (branch_it.next()) |entry| {
                try mem_keys.put(entry.key_ptr.*, {});
            }
        }

        var key_it = mem_keys.iterator();
        while (key_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const base_opt = base.memref_map.get(key);
            var fallback_value = if (base_opt) |base_state|
                base_state.value
            else blk: {
                for (branch_states) |*branch_state| {
                    if (branch_state.memref_map.get(key)) |state| break :blk state.value;
                }
                continue;
            };
            var fallback_init = if (base_opt) |base_state|
                base_state.initialized
            else
                z3.Z3_mk_false(self.context.ctx);

            var idx = branch_states.len;
            while (idx > 0) {
                idx -= 1;
                const branch_state_entry = branch_states[idx].memref_map.get(key) orelse Encoder.TrackedMemrefState{
                    .value = fallback_value,
                    .initialized = fallback_init,
                };
                const branch_val = branch_state_entry.value;
                fallback_value = if (branch_val == fallback_value)
                    branch_val
                else
                    self.encoder.encodeIte(conditions[idx], branch_val, fallback_value);
                fallback_init = self.encoder.mergeInitPredicate(conditions[idx], branch_state_entry.initialized, fallback_init);
            }

            try self.encoder.memref_map.put(key, .{
                .value = fallback_value,
                .initialized = fallback_init,
            });
        }
    }

    fn walkScfIfRegions(self: *VerificationPass, if_op: mlir.MlirOperation) anyerror!void {
        const num_regions = mlir.oraOperationGetNumRegions(if_op);
        if (num_regions == 0) return;

        const num_operands = mlir.oraOperationGetNumOperands(if_op);
        if (num_operands < 1) {
            for (0..@intCast(num_regions)) |region_idx| {
                const nested_region = mlir.oraOperationGetRegion(if_op, @intCast(region_idx));
                try self.walkMLIRRegion(nested_region);
            }
            return;
        }

        const condition_value = mlir.oraOperationGetOperand(if_op, 0);
        self.encoder.invalidateValueCaches();
        const raw_condition = try self.encoder.encodeValue(condition_value);
        const condition = self.encoder.coerceBoolean(raw_condition);
        const leaked_constraints = try self.encoder.takeConstraints(self.allocator);
        defer if (leaked_constraints.len > 0) self.allocator.free(leaked_constraints);
        const leaked_obligations = try self.encoder.takeObligations(self.allocator);
        defer if (leaked_obligations.len > 0) self.allocator.free(leaked_obligations);

        var base_state = try self.captureEncoderBranchState();
        defer base_state.deinit(self.allocator);

        var then_state = EncoderBranchState.init(self.allocator);
        defer then_state.deinit(self.allocator);
        var else_state = EncoderBranchState.init(self.allocator);
        defer else_state.deinit(self.allocator);

        // Then branch under condition.
        if (num_regions >= 1) {
            try self.restoreEncoderBranchState(&base_state);
            const saved_len = self.active_path_assumptions.items.len;
            const scoped_constraints = try self.cloneConstraintSlice(leaked_constraints);
            defer self.releaseActivePathAssumptionsFrom(saved_len);
            try self.active_path_assumptions.append(.{
                .condition = condition,
                .extra_constraints = scoped_constraints,
                .owned_extra_constraints = true,
            });
            const then_region = mlir.oraOperationGetRegion(if_op, 0);
            try self.walkMLIRRegion(then_region);
            then_state.deinit(self.allocator);
            then_state = try self.captureEncoderBranchState();
        } else {
            then_state.deinit(self.allocator);
            then_state = try self.captureEncoderBranchState();
        }

        // Else branch under !condition.
        if (num_regions >= 2) {
            try self.restoreEncoderBranchState(&base_state);
            const saved_len = self.active_path_assumptions.items.len;
            const scoped_constraints = try self.cloneConstraintSlice(leaked_constraints);
            defer self.releaseActivePathAssumptionsFrom(saved_len);
            const not_condition = self.encoder.encodeNot(condition);
            try self.active_path_assumptions.append(.{
                .condition = not_condition,
                .extra_constraints = scoped_constraints,
                .owned_extra_constraints = true,
            });
            const else_region = mlir.oraOperationGetRegion(if_op, 1);
            try self.walkMLIRRegion(else_region);
            else_state.deinit(self.allocator);
            else_state = try self.captureEncoderBranchState();
        } else {
            else_state.deinit(self.allocator);
            else_state = try self.captureEncoderBranchState();
        }

        try self.mergeEncoderBranchState(condition, &base_state, &then_state, &else_state);
    }

    fn walkConditionalReturnRegions(self: *VerificationPass, if_op: mlir.MlirOperation) anyerror!void {
        const num_operands = mlir.oraOperationGetNumOperands(if_op);
        if (num_operands < 1) {
            const num_regions = mlir.oraOperationGetNumRegions(if_op);
            for (0..@intCast(num_regions)) |region_idx| {
                const nested_region = mlir.oraOperationGetRegion(if_op, @intCast(region_idx));
                try self.walkMLIRRegion(nested_region);
            }
            return;
        }

        const condition_value = mlir.oraOperationGetOperand(if_op, 0);
        self.encoder.invalidateValueCaches();
        const raw_condition = try self.encoder.encodeValue(condition_value);
        const condition = self.encoder.coerceBoolean(raw_condition);
        const leaked_constraints = try self.encoder.takeConstraints(self.allocator);
        defer if (leaked_constraints.len > 0) self.allocator.free(leaked_constraints);
        const leaked_obligations = try self.encoder.takeObligations(self.allocator);
        defer if (leaked_obligations.len > 0) self.allocator.free(leaked_obligations);

        var base_state = try self.captureEncoderBranchState();
        defer base_state.deinit(self.allocator);

        var then_state = EncoderBranchState.init(self.allocator);
        defer then_state.deinit(self.allocator);
        var else_state = EncoderBranchState.init(self.allocator);
        defer else_state.deinit(self.allocator);

        const then_block = mlir.oraConditionalReturnOpGetThenBlock(if_op);
        if (!mlir.oraBlockIsNull(then_block)) {
            try self.restoreEncoderBranchState(&base_state);
            const saved_len = self.active_path_assumptions.items.len;
            const scoped_constraints = try self.cloneConstraintSlice(leaked_constraints);
            defer self.releaseActivePathAssumptionsFrom(saved_len);
            try self.active_path_assumptions.append(.{
                .condition = condition,
                .extra_constraints = scoped_constraints,
                .owned_extra_constraints = true,
            });

            const then_region = mlir.oraOperationGetRegion(if_op, 0);
            try self.walkMLIRRegion(then_region);
            then_state.deinit(self.allocator);
            then_state = try self.captureEncoderBranchState();
        } else {
            then_state.deinit(self.allocator);
            then_state = try self.captureEncoderBranchState();
        }

        const else_block = mlir.oraConditionalReturnOpGetElseBlock(if_op);
        if (!mlir.oraBlockIsNull(else_block)) {
            try self.restoreEncoderBranchState(&base_state);
            const saved_len = self.active_path_assumptions.items.len;
            const scoped_constraints = try self.cloneConstraintSlice(leaked_constraints);
            defer self.releaseActivePathAssumptionsFrom(saved_len);
            const not_condition = self.encoder.encodeNot(condition);
            try self.active_path_assumptions.append(.{
                .condition = not_condition,
                .extra_constraints = scoped_constraints,
                .owned_extra_constraints = true,
            });

            const else_region = mlir.oraOperationGetRegion(if_op, 1);
            try self.walkMLIRRegion(else_region);
            else_state.deinit(self.allocator);
            else_state = try self.captureEncoderBranchState();
        } else {
            else_state.deinit(self.allocator);
            else_state = try self.captureEncoderBranchState();
        }

        // Unlike scf.if, only the fallthrough (else) path reaches subsequent
        // operations in the current block. The then-region returns from the
        // enclosing function, so carrying an ite-merged state/path forward is
        // unsound for later obligations.
        try self.restoreEncoderBranchState(&else_state);
        self.encoder.invalidateValueCaches();
        const fallthrough_raw_condition = try self.encoder.encodeValue(condition_value);
        const fallthrough_condition = self.encoder.coerceBoolean(fallthrough_raw_condition);
        const fallthrough_leaked_constraints = try self.encoder.takeConstraints(self.allocator);
        defer if (fallthrough_leaked_constraints.len > 0) self.allocator.free(fallthrough_leaked_constraints);
        const fallthrough_leaked_obligations = try self.encoder.takeObligations(self.allocator);
        defer if (fallthrough_leaked_obligations.len > 0) self.allocator.free(fallthrough_leaked_obligations);

        const not_fallthrough = self.encoder.encodeNot(fallthrough_condition);
        if (astSimplifiesToBool(self.context.ctx, not_fallthrough)) |always_true| {
            if (!always_true) {
                const fallthrough_constraints = try self.cloneConstraintSlice(fallthrough_leaked_constraints);
                try self.active_path_assumptions.append(.{
                    .condition = not_fallthrough,
                    .extra_constraints = fallthrough_constraints,
                    .owned_extra_constraints = true,
                });
            }
        } else {
            const fallthrough_constraints = try self.cloneConstraintSlice(fallthrough_leaked_constraints);
            try self.active_path_assumptions.append(.{
                .condition = not_fallthrough,
                .extra_constraints = fallthrough_constraints,
                .owned_extra_constraints = true,
            });
        }
    }

    fn walkOraSwitchRegions(self: *VerificationPass, switch_op: mlir.MlirOperation) anyerror!void {
        const num_regions: usize = @intCast(mlir.oraOperationGetNumRegions(switch_op));
        if (num_regions == 0) return;

        const num_operands = mlir.oraOperationGetNumOperands(switch_op);
        if (num_operands < 1) {
            for (0..num_regions) |region_idx| {
                const nested_region = mlir.oraOperationGetRegion(switch_op, @intCast(region_idx));
                try self.walkMLIRRegion(nested_region);
            }
            return;
        }

        const scrutinee_value = mlir.oraOperationGetOperand(switch_op, 0);
        self.encoder.invalidateValueCaches();
        const scrutinee = try self.encoder.encodeValue(scrutinee_value);
        const leaked_constraints = try self.encoder.takeConstraints(self.allocator);
        defer if (leaked_constraints.len > 0) self.allocator.free(leaked_constraints);
        const leaked_obligations = try self.encoder.takeObligations(self.allocator);
        defer if (leaked_obligations.len > 0) self.allocator.free(leaked_obligations);

        var metadata = try self.getSwitchCaseMetadata(switch_op, num_regions);
        defer metadata.deinit(self.allocator);

        var base_state = try self.captureEncoderBranchState();
        defer base_state.deinit(self.allocator);

        var branch_conditions = try self.allocator.alloc(z3.Z3_ast, num_regions);
        defer self.allocator.free(branch_conditions);
        var branch_states = try self.allocator.alloc(EncoderBranchState, num_regions);
        defer {
            for (branch_states) |*branch_state| {
                branch_state.deinit(self.allocator);
            }
            self.allocator.free(branch_states);
        }
        for (branch_states) |*branch_state| {
            branch_state.* = EncoderBranchState.init(self.allocator);
        }

        var remaining = self.encoder.encodeBoolConstant(true);
        for (0..num_regions) |region_idx| {
            try self.restoreEncoderBranchState(&base_state);
            const raw_predicate = try self.buildOraSwitchCasePredicate(
                scrutinee,
                metadata.case_kinds,
                metadata.case_values,
                metadata.range_starts,
                metadata.range_ends,
                region_idx,
            );
            const effective_predicate = if (metadata.default_case_index != null and metadata.default_case_index.? == @as(i64, @intCast(region_idx)))
                remaining
            else
                self.encoder.encodeAnd(&.{ remaining, raw_predicate });
            branch_conditions[region_idx] = effective_predicate;

            const saved_len = self.active_path_assumptions.items.len;
            defer self.releaseActivePathAssumptionsFrom(saved_len);
            try self.active_path_assumptions.append(.{
                .condition = effective_predicate,
                .extra_constraints = try self.cloneConstraintSlice(leaked_constraints),
                .owned_extra_constraints = true,
            });
            const region = mlir.oraOperationGetRegion(switch_op, @intCast(region_idx));
            try self.walkMLIRRegion(region);
            branch_states[region_idx].deinit(self.allocator);
            branch_states[region_idx] = try self.captureEncoderBranchState();

            if (metadata.default_case_index == null or metadata.default_case_index.? != @as(i64, @intCast(region_idx))) {
                remaining = self.encoder.encodeAnd(&.{ remaining, self.encoder.encodeNot(raw_predicate) });
            }
        }

        try self.mergeEncoderBranchStatesMany(branch_conditions, &base_state, branch_states);
    }

    fn getSwitchCaseAttrValues(self: *VerificationPass, op: mlir.MlirOperation, name: []const u8) ![]i64 {
        const attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate(name.ptr, name.len));
        if (mlir.oraAttributeIsNull(attr)) return try self.allocator.alloc(i64, 0);

        const count: usize = @intCast(mlir.oraArrayAttrGetNumElements(attr));
        var values = try self.allocator.alloc(i64, count);
        for (0..count) |i| {
            values[i] = mlir.oraIntegerAttrGetValueSInt(mlir.oraArrayAttrGetElement(attr, i));
        }
        return values;
    }

    fn getSwitchDefaultCaseIndex(_: *VerificationPass, op: mlir.MlirOperation) ?i64 {
        const attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("default_case_index".ptr, "default_case_index".len));
        if (mlir.oraAttributeIsNull(attr)) return null;
        return mlir.oraIntegerAttrGetValueSInt(attr);
    }

    fn getSwitchCaseMetadata(self: *VerificationPass, op: mlir.MlirOperation, num_regions: usize) !SwitchCaseMetadata {
        var metadata = SwitchCaseMetadata{
            .case_kinds = try self.allocator.alloc(i64, num_regions),
            .case_values = try self.allocator.alloc(i64, num_regions),
            .range_starts = try self.allocator.alloc(i64, num_regions),
            .range_ends = try self.allocator.alloc(i64, num_regions),
            .default_case_index = null,
        };
        errdefer metadata.deinit(self.allocator);

        @memset(metadata.case_kinds, 0);
        @memset(metadata.case_values, 0);
        @memset(metadata.range_starts, 0);
        @memset(metadata.range_ends, 0);

        const attr_case_kinds = try self.getSwitchCaseAttrValues(op, "case_kinds");
        defer self.allocator.free(attr_case_kinds);
        const attr_case_values = try self.getSwitchCaseAttrValues(op, "case_values");
        defer self.allocator.free(attr_case_values);
        const attr_range_starts = try self.getSwitchCaseAttrValues(op, "range_starts");
        defer self.allocator.free(attr_range_starts);
        const attr_range_ends = try self.getSwitchCaseAttrValues(op, "range_ends");
        defer self.allocator.free(attr_range_ends);
        const attr_default = self.getSwitchDefaultCaseIndex(op);

        if (attr_case_kinds.len >= num_regions) {
            @memcpy(metadata.case_kinds, attr_case_kinds[0..num_regions]);
            if (attr_case_values.len >= num_regions) @memcpy(metadata.case_values, attr_case_values[0..num_regions]);
            if (attr_range_starts.len >= num_regions) @memcpy(metadata.range_starts, attr_range_starts[0..num_regions]);
            if (attr_range_ends.len >= num_regions) @memcpy(metadata.range_ends, attr_range_ends[0..num_regions]);
            metadata.default_case_index = attr_default;
            return metadata;
        }

        try self.parseSwitchCaseMetadataFromPrint(op, &metadata);
        return metadata;
    }

    fn parseSwitchCaseMetadataFromPrint(self: *VerificationPass, op: mlir.MlirOperation, metadata: *SwitchCaseMetadata) !void {
        const printed = mlir.oraOperationPrintToString(op);
        defer if (printed.data != null) @import("mlir_c_api").freeStringRef(printed);
        if (printed.data == null or printed.length == 0) return error.UnsupportedOperation;

        var case_index: usize = 0;
        var lines = std.mem.splitScalar(u8, printed.data[0..printed.length], '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (std.mem.startsWith(u8, trimmed, "case ")) {
                if (case_index >= metadata.case_kinds.len) break;
                const body = trimmed["case ".len..];
                const arrow_index = std.mem.indexOf(u8, body, "=>") orelse continue;
                const pattern = std.mem.trim(u8, body[0..arrow_index], " \t");
                if (std.mem.indexOf(u8, pattern, "...")) |range_index| {
                    metadata.case_kinds[case_index] = 1;
                    metadata.range_starts[case_index] = try self.parseSwitchPatternInt(pattern[0..range_index]);
                    metadata.range_ends[case_index] = try self.parseSwitchPatternInt(pattern[range_index + 3 ..]);
                } else {
                    metadata.case_kinds[case_index] = 0;
                    metadata.case_values[case_index] = try self.parseSwitchPatternInt(pattern);
                }
                case_index += 1;
            } else if (std.mem.startsWith(u8, trimmed, "else =>")) {
                if (case_index >= metadata.case_kinds.len) break;
                metadata.case_kinds[case_index] = 2;
                metadata.default_case_index = @intCast(case_index);
                case_index += 1;
            }
        }
        if (case_index < metadata.case_kinds.len) return error.UnsupportedOperation;
    }

    fn parseSwitchPatternInt(_: *VerificationPass, text: []const u8) !i64 {
        const trimmed = std.mem.trim(u8, text, " \t");
        if (std.mem.eql(u8, trimmed, "true")) return 1;
        if (std.mem.eql(u8, trimmed, "false")) return 0;
        return std.fmt.parseInt(i64, trimmed, 10) catch error.UnsupportedOperation;
    }

    fn buildOraSwitchCasePredicate(
        self: *VerificationPass,
        scrutinee: z3.Z3_ast,
        case_kinds: []const i64,
        case_values: []const i64,
        range_starts: []const i64,
        range_ends: []const i64,
        case_index: usize,
    ) !z3.Z3_ast {
        if (case_index >= case_kinds.len) return error.UnsupportedOperation;
        const sort = z3.Z3_get_sort(self.context.ctx, scrutinee);
        return switch (case_kinds[case_index]) {
            0 => blk: {
                if (case_index >= case_values.len) return error.UnsupportedOperation;
                const case_value = try self.encoder.encodeScalarValueForSort(case_values[case_index], sort);
                break :blk self.encoder.encodeComparisonOp(.Eq, scrutinee, case_value);
            },
            1 => blk: {
                if (case_index >= range_starts.len or case_index >= range_ends.len) return error.UnsupportedOperation;
                const start_value = try self.encoder.encodeScalarValueForSort(range_starts[case_index], sort);
                const end_value = try self.encoder.encodeScalarValueForSort(range_ends[case_index], sort);
                const lower = self.encoder.encodeComparisonOp(.Ge, scrutinee, start_value);
                const upper = self.encoder.encodeComparisonOp(.Le, scrutinee, end_value);
                break :blk self.encoder.encodeAnd(&.{ lower, upper });
            },
            2 => self.encoder.encodeBoolConstant(true),
            else => error.UnsupportedOperation,
        };
    }

    fn collectRegionPredecessorCounts(
        _: *VerificationPass,
        region: mlir.MlirRegion,
        counts: *std.AutoHashMap(usize, usize),
    ) !void {
        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) {
            const term = mlir.oraBlockGetTerminator(block);
            if (!mlir.oraOperationIsNull(term)) {
                const num_succ = mlir.mlirOperationGetNumSuccessors(term);
                for (0..@intCast(num_succ)) |succ_idx| {
                    const succ_block = mlir.mlirOperationGetSuccessor(term, @intCast(succ_idx));
                    if (mlir.oraBlockIsNull(succ_block)) continue;
                    const succ_key = @intFromPtr(succ_block.ptr);
                    const prev = counts.get(succ_key) orelse 0;
                    try counts.put(succ_key, prev + 1);
                }
            }
            block = mlir.oraBlockGetNextInRegion(block);
        }
    }

    /// Process a single MLIR operation to extract verification annotations
    fn processMLIROperation(self: *VerificationPass, op: mlir.MlirOperation) !?usize {
        const op_name_ref = self.getMLIROperationName(op);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];

        if (!std.mem.eql(u8, op_name, "func.func") and
            self.current_function_name != null and
            !self.current_function_verify_enabled)
        {
            return null;
        }

        if (self.filter_function_name) |target_fn| {
            const current = self.current_function_name orelse return null;
            if (!std.mem.eql(u8, current, target_fn)) return null;
        }

        try self.observeStateOperation(op, op_name);

        // check for verification operations
        if (std.mem.eql(u8, op_name, "ora.requires")) {
            // extract requires condition
            // get the condition operand (should be the first and only operand)
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                _ = try self.recordEncodedAnnotation(op, .Requires, condition_value, null);
            }
        } else if (std.mem.eql(u8, op_name, "ora.ensures")) {
            // extract ensures condition
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                _ = try self.recordEncodedAnnotation(op, .Ensures, condition_value, null);
            }
        } else if (std.mem.eql(u8, op_name, "ora.invariant")) {
            // extract invariant condition
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                const invariant_kind: AnnotationKind = if (self.findEnclosingLoopOp(op) != null)
                    .LoopInvariant
                else
                    .ContractInvariant;
                return try self.recordEncodedAnnotation(op, invariant_kind, condition_value, null);
            }
        } else if (std.mem.eql(u8, op_name, "ora.refinement_guard")) {
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                const guard_id = try self.getStringAttr(op, "ora.guard_id", &self.guard_id_storage);
                _ = try self.recordEncodedAnnotation(op, .RefinementGuard, condition_value, guard_id);
            }
        } else if (std.mem.eql(u8, op_name, "ora.assume")) {
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                const origin_attr = try self.getStringAttr(op, "ora.assume_origin", &self.guard_id_storage);
                const assume_kind: AnnotationKind = if (origin_attr) |origin| blk: {
                    if (std.mem.eql(u8, origin, "path")) break :blk .PathAssume;
                    break :blk .Assume;
                } else blk: {
                    const context_attr = try self.getStringAttr(op, "ora.verification_context", &self.guard_id_storage);
                    if (context_attr) |context_str| {
                        if (std.mem.eql(u8, context_str, "path_assumption")) break :blk .PathAssume;
                    }
                    break :blk .Assume;
                };
                const annotation_index = try self.recordEncodedAnnotation(op, assume_kind, condition_value, null);
                if (assume_kind == .PathAssume) {
                    const ann = self.encoded_annotations.items[annotation_index];
                    try self.active_path_assumptions.append(.{
                        .condition = ann.condition,
                        .extra_constraints = ann.extra_constraints,
                    });
                }
            }
        } else if (std.mem.eql(u8, op_name, "cf.assert") or std.mem.eql(u8, op_name, "ora.assert")) {
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                const guard_id = try self.getStringAttr(op, "ora.guard_id", &self.guard_id_storage);
                var tagged_assert = false;
                const requires_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("ora.requires", 12));
                if (!mlir.oraAttributeIsNull(requires_attr)) {
                    _ = try self.recordEncodedAnnotation(op, .Requires, condition_value, null);
                    tagged_assert = true;
                }
                const ensures_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("ora.ensures", 11));
                if (!mlir.oraAttributeIsNull(ensures_attr)) {
                    _ = try self.recordEncodedAnnotation(op, .Ensures, condition_value, null);
                    tagged_assert = true;
                }
                if (!tagged_assert and self.verify_mode == .Full) {
                    const verification_type_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("ora.verification_type", 21));
                    const obligation_kind: AnnotationKind = if (!mlir.oraAttributeIsNull(verification_type_attr)) blk: {
                        const ref = mlir.oraStringAttrGetValue(verification_type_attr);
                        if (ref.data != null and ref.length > 0 and std.mem.eql(u8, ref.data[0..ref.length], "guard")) {
                            break :blk .Guard;
                        }
                        break :blk .ContractInvariant;
                    } else .ContractInvariant;
                    // In full mode, treat untagged assert ops as proof obligations.
                    _ = try self.recordEncodedAnnotation(op, obligation_kind, condition_value, guard_id);
                }
            }
        }
        return null;
    }

    fn observeStateOperation(self: *VerificationPass, op: mlir.MlirOperation, op_name: []const u8) !void {
        const should_observe =
            std.mem.eql(u8, op_name, "memref.alloca") or
            std.mem.eql(u8, op_name, "memref.store") or
            std.mem.eql(u8, op_name, "func.call") or
            std.mem.eql(u8, op_name, "call") or
            std.mem.eql(u8, op_name, "ora.sstore") or
            std.mem.eql(u8, op_name, "ora.tstore") or
            std.mem.eql(u8, op_name, "ora.map_store");
        if (!should_observe) return;

        _ = try self.encoder.encodeOperation(op);

        const leaked_constraints = try self.encoder.takeConstraints(self.allocator);
        defer if (leaked_constraints.len > 0) self.allocator.free(leaked_constraints);
        const leaked_obligations = try self.encoder.takeObligations(self.allocator);
        defer if (leaked_obligations.len > 0) self.allocator.free(leaked_obligations);

        // Preserve obligations discovered while observing stateful ops (notably
        // call summaries) so public entrypoints with no explicit requires/asserts
        // still prove checked arithmetic safety in reachable callees.
        if (leaked_obligations.len > 0) {
            const function_name = self.current_function_name orelse "unknown";
            const loc = try self.getLocationInfo(op);
            const path_constraints = try self.captureActivePathConstraints();
            const loop_owner = if (self.findEnclosingLoopOp(op)) |loop_op| @as(?u64, @intFromPtr(loop_op.ptr)) else null;
            var loop_step_condition: ?z3.Z3_ast = null;
            var loop_step_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
            defer if (path_constraints.len > 0) self.allocator.free(path_constraints);
            defer if (loop_step_extra_constraints.len > 0) self.allocator.free(loop_step_extra_constraints);
            const path_guard = if (path_constraints.len > 0)
                self.encoder.encodeAnd(path_constraints)
            else
                null;
            if (loop_owner != null) {
                loop_step_condition = try self.encodeLoopContinueCondition(op);
                loop_step_extra_constraints = try self.encoder.takeConstraints(self.allocator);
            }

            for (leaked_obligations) |obligation| {
                const guarded_obligation = if (path_guard) |guard|
                    self.encoder.encodeImplies(guard, obligation)
                else
                    obligation;
                try appendEncodedAnnotationUnique(self, .{
                    .function_name = function_name,
                    .kind = .ContractInvariant,
                    .condition = guarded_obligation,
                    .extra_constraints = try self.cloneConstraintSlice(leaked_constraints),
                    .path_constraints = try self.cloneConstraintSlice(path_constraints),
                    .old_condition = null,
                    .old_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_step_condition = loop_step_condition,
                    .loop_step_extra_constraints = try self.cloneConstraintSlice(loop_step_extra_constraints),
                    .loop_exit_condition = null,
                    .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                    .file = loc.file,
                    .line = loc.line,
                    .column = loc.column,
                    .guard_id = null,
                    .loop_owner = loop_owner,
                });
            }
        }

        // State observation should not leak ad-hoc constraints into subsequent
        // annotation queries.
        // Encoding state ops can cache intermediate value encodings (e.g.,
        // ora.struct_init) whose defining constraints were just discarded.
        // Clear expression caches so later annotation encoding re-materializes
        // those constraints instead of reusing under-constrained ASTs.
        self.encoder.invalidateValueCaches();
    }

    /// Get MLIR operation name as string
    fn getMLIROperationName(_: *VerificationPass, op: mlir.MlirOperation) mlir.MlirStringRef {
        return mlir.oraOperationGetName(op);
    }

    fn getFunctionNameFromOp(self: *VerificationPass, op: mlir.MlirOperation) !?[]const u8 {
        const name_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("sym_name", 8));
        if (mlir.oraAttributeIsNull(name_attr)) return null;
        const name_ref = mlir.oraStringAttrGetValue(name_attr);
        if (name_ref.data == null or name_ref.length == 0) return null;
        const name_slice = name_ref.data[0..name_ref.length];
        const dup = try self.allocator.dupe(u8, name_slice);
        try self.function_name_storage.append(dup);
        return dup;
    }

    fn getStringAttr(self: *VerificationPass, op: mlir.MlirOperation, name: []const u8, storage: *ManagedArrayList([]const u8)) !?[]const u8 {
        const attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate(name.ptr, name.len));
        if (mlir.oraAttributeIsNull(attr)) return null;
        const value_ref = mlir.oraStringAttrGetValue(attr);
        if (value_ref.data == null or value_ref.length == 0) return null;
        const value_slice = value_ref.data[0..value_ref.length];
        const dup = try self.allocator.dupe(u8, value_slice);
        try storage.append(dup);
        return dup;
    }

    fn shouldVerifyFunctionOp(_: *VerificationPass, op: mlir.MlirOperation) bool {
        const visibility_attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate("ora.visibility", 14));
        if (mlir.oraAttributeIsNull(visibility_attr)) return true;
        const visibility_ref = mlir.oraStringAttrGetValue(visibility_attr);
        if (visibility_ref.data == null or visibility_ref.length == 0) return true;
        const visibility = visibility_ref.data[0..visibility_ref.length];
        return std.mem.eql(u8, visibility, "pub") or
            std.mem.eql(u8, visibility, "public") or
            std.mem.eql(u8, visibility, "external");
    }

    fn recordEncodedAnnotation(
        self: *VerificationPass,
        op: mlir.MlirOperation,
        kind: AnnotationKind,
        condition_value: mlir.MlirValue,
        guard_id: ?[]const u8,
    ) !usize {
        const function_name = self.current_function_name orelse "unknown";
        const loop_owner = if (self.findEnclosingLoopOp(op)) |loop_op| @as(?u64, @intFromPtr(loop_op.ptr)) else null;
        const failing_op = if (mlir.oraValueIsAOpResult(condition_value))
            mlir.oraOpResultGetOwner(condition_value)
        else
            op;
        var base_encoder_state = try self.captureEncoderBranchState();
        defer base_encoder_state.deinit(self.allocator);
        defer self.restoreEncoderBranchState(&base_encoder_state) catch {};

        const encoded = blk: {
            if (self.encoder.tryEncodeAssertCondition(op, .Current) catch |err| {
                self.encoder.noteDegradationAtOp(failing_op, "unsupported annotation condition");
                return err;
            }) |specialized| {
                break :blk specialized;
            }
            break :blk self.encoder.encodeValue(condition_value) catch |err| {
                self.encoder.noteDegradationAtOp(failing_op, "unsupported annotation condition");
                return err;
            };
        };
        const raw_extra_constraints = try self.encoder.takeConstraints(self.allocator);
        defer if (raw_extra_constraints.len > 0) self.allocator.free(raw_extra_constraints);
        const safety_obligations = try self.encoder.takeObligations(self.allocator);
        defer if (safety_obligations.len > 0) self.allocator.free(safety_obligations);
        const path_constraints = try self.captureActivePathConstraints();
        const path_guard = if (path_constraints.len > 0)
            self.encoder.encodeAnd(path_constraints)
        else
            null;

        var extra_constraint_list = ManagedArrayList(z3.Z3_ast).init(self.allocator);
        defer extra_constraint_list.deinit();
        try addConstraintSlice(&extra_constraint_list, raw_extra_constraints);
        try addExplicitOldLinkConstraintsForAst(self, &extra_constraint_list, encoded);
        const extra_constraints = try extra_constraint_list.toOwnedSlice();

        var old_condition: ?z3.Z3_ast = null;
        var old_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
        var loop_entry_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
        var loop_step_condition: ?z3.Z3_ast = null;
        var loop_step_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
        var loop_exit_condition: ?z3.Z3_ast = null;
        var loop_exit_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
        if (loop_owner != null) {
            try self.mergeStableCurrentValueCachesIntoSnapshot(&base_encoder_state);
            try self.restoreEncoderBranchState(&base_encoder_state);
            loop_step_condition = try self.encodeLoopContinueCondition(op);
            loop_step_extra_constraints = try self.encoder.takeConstraints(self.allocator);
        }
        if (kind == .LoopInvariant) {
            try self.restoreEncoderBranchState(&base_encoder_state);
            old_condition = if (try self.encoder.tryEncodeAssertCondition(op, .Old)) |specialized_old|
                specialized_old
            else
                try self.encoder.encodeValueOld(condition_value);
            old_extra_constraints = try self.encoder.takeConstraints(self.allocator);
            loop_entry_extra_constraints = try self.encodeLoopEntryConstraints(op);
            const old_safety = try self.encoder.takeObligations(self.allocator);
            defer if (old_safety.len > 0) self.allocator.free(old_safety);
            for (old_safety) |obligation| {
                const loc_old = try self.getLocationInfo(op);
                try self.encoded_annotations.append(.{
                    .function_name = function_name,
                    .kind = .ContractInvariant,
                    .condition = obligation,
                    .extra_constraints = &[_]z3.Z3_ast{},
                    .path_constraints = try self.cloneConstraintSlice(path_constraints),
                    .old_condition = null,
                    .old_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_entry_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_step_condition = null,
                    .loop_step_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_exit_condition = null,
                    .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                    .file = loc_old.file,
                    .line = loc_old.line,
                    .column = loc_old.column,
                    .guard_id = null,
                    .loop_owner = loop_owner,
                });
            }
            try self.restoreEncoderBranchState(&base_encoder_state);
            loop_exit_condition = try self.encodeLoopExitCondition(op);
            loop_exit_extra_constraints = try self.encoder.takeConstraints(self.allocator);
            const loop_exit_safety = try self.encoder.takeObligations(self.allocator);
            defer if (loop_exit_safety.len > 0) self.allocator.free(loop_exit_safety);
            for (loop_exit_safety) |obligation| {
                const loc_loop = try self.getLocationInfo(op);
                try self.encoded_annotations.append(.{
                    .function_name = function_name,
                    .kind = .ContractInvariant,
                    .condition = obligation,
                    .extra_constraints = &[_]z3.Z3_ast{},
                    .path_constraints = try self.cloneConstraintSlice(path_constraints),
                    .old_condition = null,
                    .old_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_entry_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_step_condition = null,
                    .loop_step_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_exit_condition = null,
                    .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                    .file = loc_loop.file,
                    .line = loc_loop.line,
                    .column = loc_loop.column,
                    .guard_id = null,
                    .loop_owner = loop_owner,
                });
            }
        }

        const loc = try self.getLocationInfo(op);
        const annotation_label = try self.annotationLabelForOp(op, kind);
        const annotation_index = self.encoded_annotations.items.len;
        try self.encoded_annotations.append(.{
            .function_name = function_name,
            .kind = kind,
            .condition = encoded,
            .condition_value = condition_value,
            .source_op = op,
            .extra_constraints = extra_constraints,
            .path_constraints = path_constraints,
            .old_condition = old_condition,
            .old_extra_constraints = old_extra_constraints,
            .loop_entry_extra_constraints = loop_entry_extra_constraints,
            .loop_step_condition = loop_step_condition,
            .loop_step_extra_constraints = loop_step_extra_constraints,
            .loop_step_head_condition = null,
            .loop_step_head_extra_constraints = &[_]z3.Z3_ast{},
            .loop_step_body_condition = null,
            .loop_step_body_extra_constraints = &[_]z3.Z3_ast{},
            .loop_exit_condition = loop_exit_condition,
            .loop_exit_extra_constraints = loop_exit_extra_constraints,
            .file = loc.file,
            .line = loc.line,
            .column = loc.column,
            .label = annotation_label,
            .guard_id = guard_id,
            .loop_owner = loop_owner,
        });

        // Safety obligations emitted by the encoder (e.g. non-zero divisor,
        // multiplication overflow checks) are tracked as contract invariants.
        for (safety_obligations) |obligation| {
            const obligation_extra_constraints = try self.cloneConstraintSlice(extra_constraints);
            const guarded_obligation = if (path_guard) |guard|
                self.encoder.encodeImplies(guard, obligation)
            else
                obligation;
            try appendEncodedAnnotationUnique(self, .{
                .function_name = function_name,
                .kind = .ContractInvariant,
                .condition = guarded_obligation,
                .condition_value = null,
                .source_op = null,
                .extra_constraints = obligation_extra_constraints,
                .path_constraints = try self.cloneConstraintSlice(path_constraints),
                .old_condition = null,
                .old_extra_constraints = &[_]z3.Z3_ast{},
                .loop_step_condition = loop_step_condition,
                .loop_step_extra_constraints = try self.cloneConstraintSlice(loop_step_extra_constraints),
                .loop_step_head_condition = null,
                .loop_step_head_extra_constraints = &[_]z3.Z3_ast{},
                .loop_step_body_condition = null,
                .loop_step_body_extra_constraints = &[_]z3.Z3_ast{},
                .loop_exit_condition = null,
                .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                .file = loc.file,
                .line = loc.line,
                .column = loc.column,
                .label = annotation_label,
                .guard_id = null,
                .loop_owner = loop_owner,
            });
        }

        try self.mergeStableCurrentValueCachesIntoSnapshot(&base_encoder_state);

        // Avoid cross-annotation reuse of cached expression ASTs whose defining
        // side-constraints were already consumed by takeConstraints() above.
        // This is especially important for old(...) encodings in postconditions:
        // each annotation query must re-materialize the old/current linkage.
        self.encoder.invalidateValueCaches();

        return annotation_index;
    }

    fn captureActivePathConstraints(self: *VerificationPass) ![]const z3.Z3_ast {
        if (self.active_path_assumptions.items.len == 0) {
            return &[_]z3.Z3_ast{};
        }
        var constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
        defer constraints.deinit();

        for (self.active_path_assumptions.items) |assume| {
            for (assume.extra_constraints) |extra| {
                try self.appendNormalizedPathConstraint(&constraints, extra);
            }
            try self.appendNormalizedPathConstraint(&constraints, assume.condition);
        }

        return try constraints.toOwnedSlice();
    }

    fn appendNormalizedPathConstraint(
        self: *VerificationPass,
        constraints: *ManagedArrayList(z3.Z3_ast),
        condition: z3.Z3_ast,
    ) !void {
        const simplified = z3.Z3_simplify(self.context.ctx, condition);
        if (astSimplifiesToBool(self.context.ctx, simplified)) |value| {
            if (!value and !constraintSliceContains(self, constraints.items, simplified)) {
                try constraints.append(simplified);
            }
            return;
        }

        if (z3.Z3_get_ast_kind(self.context.ctx, simplified) != z3.Z3_APP_AST) {
            if (!constraintSliceContains(self, constraints.items, simplified)) {
                try constraints.append(simplified);
            }
            return;
        }

        const app = z3.Z3_to_app(self.context.ctx, simplified);
        const decl = z3.Z3_get_app_decl(self.context.ctx, app);
        const kind = z3.Z3_get_decl_kind(self.context.ctx, decl);

        if (kind == z3.Z3_OP_ITE and z3.Z3_get_app_num_args(self.context.ctx, app) == 3) {
            const ite_cond = z3.Z3_get_app_arg(self.context.ctx, app, 0);
            const ite_then = z3.Z3_get_app_arg(self.context.ctx, app, 1);
            const ite_else = z3.Z3_get_app_arg(self.context.ctx, app, 2);

            const then_bool = astSimplifiesToBool(self.context.ctx, ite_then);
            const else_bool = astSimplifiesToBool(self.context.ctx, ite_else);

            if (then_bool) |then_value| {
                if (else_bool) |else_value| {
                    if (then_value and !else_value) {
                        try self.appendNormalizedPathConstraint(constraints, ite_cond);
                        return;
                    }
                    if (!then_value and else_value) {
                        try self.appendNormalizedPathConstraint(constraints, z3.Z3_mk_not(self.context.ctx, ite_cond));
                        return;
                    }
                } else if (!then_value) {
                    // ite c false e == !c && e
                    try self.appendNormalizedPathConstraint(constraints, z3.Z3_mk_not(self.context.ctx, ite_cond));
                    try self.appendNormalizedPathConstraint(constraints, ite_else);
                    return;
                } else {
                    // ite c true e == c || e
                    try self.appendNormalizedPathConstraint(constraints, z3.Z3_mk_or(
                        self.context.ctx,
                        2,
                        &[_]z3.Z3_ast{ ite_cond, ite_else },
                    ));
                    return;
                }
            }

            if (else_bool) |else_value| {
                if (!else_value) {
                    // ite c t false == c && t
                    try self.appendNormalizedPathConstraint(constraints, ite_cond);
                    try self.appendNormalizedPathConstraint(constraints, ite_then);
                    return;
                } else {
                    // ite c t true == !c || t
                    try self.appendNormalizedPathConstraint(constraints, z3.Z3_mk_or(
                        self.context.ctx,
                        2,
                        &[_]z3.Z3_ast{ z3.Z3_mk_not(self.context.ctx, ite_cond), ite_then },
                    ));
                    return;
                }
            }
        }

        if (kind == z3.Z3_OP_AND) {
            const num_args = z3.Z3_get_app_num_args(self.context.ctx, app);
            for (0..@intCast(num_args)) |arg_idx| {
                try self.appendNormalizedPathConstraint(constraints, z3.Z3_get_app_arg(self.context.ctx, app, @intCast(arg_idx)));
            }
            return;
        }

        if (kind == z3.Z3_OP_NOT and z3.Z3_get_app_num_args(self.context.ctx, app) == 1) {
            const inner = z3.Z3_get_app_arg(self.context.ctx, app, 0);
            if (z3.Z3_get_ast_kind(self.context.ctx, inner) == z3.Z3_APP_AST) {
                const inner_app = z3.Z3_to_app(self.context.ctx, inner);
                const inner_decl = z3.Z3_get_app_decl(self.context.ctx, inner_app);
                if (z3.Z3_get_decl_kind(self.context.ctx, inner_decl) == z3.Z3_OP_ITE and
                    z3.Z3_get_app_num_args(self.context.ctx, inner_app) == 3)
                {
                    const ite_cond = z3.Z3_get_app_arg(self.context.ctx, inner_app, 0);
                    const ite_then = z3.Z3_get_app_arg(self.context.ctx, inner_app, 1);
                    const ite_else = z3.Z3_get_app_arg(self.context.ctx, inner_app, 2);

                    if (astSimplifiesToBool(self.context.ctx, ite_then)) |then_value| {
                        if (then_value) {
                            // not(ite c true e)  ==  !c && !e
                            try self.appendNormalizedPathConstraint(constraints, z3.Z3_mk_not(self.context.ctx, ite_cond));
                            try self.appendNormalizedPathConstraint(constraints, z3.Z3_mk_not(self.context.ctx, ite_else));
                            return;
                        }
                    }
                    if (astSimplifiesToBool(self.context.ctx, ite_else)) |else_value| {
                        if (else_value) {
                            // not(ite c t true) == c && !t
                            try self.appendNormalizedPathConstraint(constraints, ite_cond);
                            try self.appendNormalizedPathConstraint(constraints, z3.Z3_mk_not(self.context.ctx, ite_then));
                            return;
                        }
                    }
                }
            }
        }

        if (shouldSkipHeavyPathConstraint(self.context.ctx, simplified)) {
            return;
        }

        if (!constraintSliceContains(self, constraints.items, simplified)) {
            try constraints.append(simplified);
        }
    }

    fn cloneConstraintSlice(self: *VerificationPass, constraints: []const z3.Z3_ast) ![]const z3.Z3_ast {
        if (constraints.len == 0) {
            return &[_]z3.Z3_ast{};
        }
        return try self.allocator.dupe(z3.Z3_ast, constraints);
    }

    fn encodeLoopEntryConstraints(self: *VerificationPass, invariant_op: mlir.MlirOperation) ![]const z3.Z3_ast {
        const loop_op = self.findEnclosingLoopOp(invariant_op) orelse return &[_]z3.Z3_ast{};
        const loop_name_ref = mlir.oraOperationGetName(loop_op);
        defer @import("mlir_c_api").freeStringRef(loop_name_ref);
        const loop_name = if (loop_name_ref.data == null or loop_name_ref.length == 0)
            ""
        else
            loop_name_ref.data[0..loop_name_ref.length];

        if (!std.mem.eql(u8, loop_name, "scf.while")) {
            return &[_]z3.Z3_ast{};
        }

        const before_block = mlir.oraScfWhileOpGetBeforeBlock(loop_op);
        if (mlir.oraBlockIsNull(before_block)) return &[_]z3.Z3_ast{};

        var constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
        defer constraints.deinit();

        const num_init_operands = mlir.oraOperationGetNumOperands(loop_op);
        var i: usize = 0;
        while (i < num_init_operands and i < mlir.oraBlockGetNumArguments(before_block)) : (i += 1) {
            const before_arg = mlir.oraBlockGetArgument(before_block, i);
            const init_operand = mlir.oraOperationGetOperand(loop_op, i);
            const before_ast = try self.encoder.encodeValue(before_arg);
            const init_ast = try self.encoder.encodeValue(init_operand);
            const pending = try self.encoder.takeConstraints(self.allocator);
            defer if (pending.len > 0) self.allocator.free(pending);
            try addConstraintSlice(&constraints, pending);
            try constraints.append(z3.Z3_mk_eq(self.context.ctx, before_ast, init_ast));
        }

        return try constraints.toOwnedSlice();
    }

    fn refreshLoopInvariantStepConditions(
        self: *VerificationPass,
        annotation_indices: []const usize,
    ) !void {
        var base_encoder_state = try self.captureEncoderBranchState();
        defer base_encoder_state.deinit(self.allocator);

        for (annotation_indices) |annotation_index| {
            if (annotation_index >= self.encoded_annotations.items.len) continue;
            var ann = &self.encoded_annotations.items[annotation_index];
            if (ann.kind != .LoopInvariant) continue;
            const source_op = ann.source_op orelse continue;
            const condition_value = ann.condition_value orelse continue;
            const owner_block = mlir.mlirOperationGetBlock(source_op);
            if (mlir.oraBlockIsNull(owner_block)) continue;
            const parent_op = mlir.mlirBlockGetParentOperation(owner_block);
            if (mlir.oraOperationIsNull(parent_op)) continue;
            const parent_name_ref = mlir.oraOperationGetName(parent_op);
            defer @import("mlir_c_api").freeStringRef(parent_name_ref);
            const parent_name = if (parent_name_ref.data == null or parent_name_ref.length == 0)
                ""
            else
                parent_name_ref.data[0..parent_name_ref.length];
            if (!std.mem.eql(u8, parent_name, "scf.while")) continue;

            const after_block = mlir.oraScfWhileOpGetAfterBlock(parent_op);
            if (mlir.oraBlockIsNull(after_block) or after_block.ptr != owner_block.ptr) continue;
            const terminator = mlir.oraBlockGetTerminator(owner_block);
            if (mlir.oraOperationIsNull(terminator)) continue;
            const term_name_ref = mlir.oraOperationGetName(terminator);
            defer @import("mlir_c_api").freeStringRef(term_name_ref);
            const term_name = if (term_name_ref.data == null or term_name_ref.length == 0)
                ""
            else
                term_name_ref.data[0..term_name_ref.length];
            if (!std.mem.eql(u8, term_name, "scf.yield")) continue;

            try self.restoreEncoderBranchState(&base_encoder_state);
            self.clearEncoderGlobalMap();
            self.encoder.memref_map.clearRetainingCapacity();
            self.encoder.value_map.clearRetainingCapacity();
            self.encoder.value_map_old.clearRetainingCapacity();
            self.encoder.value_bindings.clearRetainingCapacity();
            self.encoder.materialized_calls.clearRetainingCapacity();
            {
                var written_it = self.encoder.written_global_slots.iterator();
                while (written_it.next()) |entry| {
                    self.allocator.free(entry.key_ptr.*);
                }
                self.encoder.written_global_slots.clearRetainingCapacity();
            }

            const after_arg_count: usize = @intCast(mlir.oraBlockGetNumArguments(owner_block));
            var i: usize = 0;
            while (i < after_arg_count) : (i += 1) {
                const after_arg = mlir.oraBlockGetArgument(owner_block, @intCast(i));
                const symbolic_arg = try self.encoder.encodeValue(after_arg);
                const transient_constraints = try self.encoder.takeConstraints(self.allocator);
                defer if (transient_constraints.len > 0) self.allocator.free(transient_constraints);
                try self.encoder.bindValue(after_arg, symbolic_arg);
            }

            var prefix_op = mlir.oraBlockGetFirstOperation(owner_block);
            while (!mlir.oraOperationIsNull(prefix_op) and prefix_op.ptr != source_op.ptr) : (prefix_op = mlir.oraOperationGetNextInBlock(prefix_op)) {
                self.encoder.encodeStateEffectsInOperation(prefix_op);
                const transient_constraints = try self.encoder.takeConstraints(self.allocator);
                defer if (transient_constraints.len > 0) self.allocator.free(transient_constraints);
                const transient_obligations = try self.encoder.takeObligations(self.allocator);
                defer if (transient_obligations.len > 0) self.allocator.free(transient_obligations);
            }

            const head_condition = blk: {
                if (try self.encoder.tryEncodeAssertCondition(source_op, .Current)) |specialized| {
                    break :blk specialized;
                }
                break :blk try self.encoder.encodeValue(condition_value);
            };
            var head_constraint_list = ManagedArrayList(z3.Z3_ast).init(self.allocator);
            defer head_constraint_list.deinit();
            const raw_head_constraints = try self.encoder.takeConstraints(self.allocator);
            defer if (raw_head_constraints.len > 0) self.allocator.free(raw_head_constraints);
            try addConstraintSlice(&head_constraint_list, raw_head_constraints);

            var body_op = mlir.oraOperationGetNextInBlock(source_op);
            while (!mlir.oraOperationIsNull(body_op) and body_op.ptr != terminator.ptr) : (body_op = mlir.oraOperationGetNextInBlock(body_op)) {
                self.encoder.encodeStateEffectsInOperation(body_op);
                const transient_constraints = try self.encoder.takeConstraints(self.allocator);
                defer if (transient_constraints.len > 0) self.allocator.free(transient_constraints);
                const transient_obligations = try self.encoder.takeObligations(self.allocator);
                defer if (transient_obligations.len > 0) self.allocator.free(transient_obligations);
            }

            const bind_count = @min(
                after_arg_count,
                @as(usize, @intCast(mlir.oraOperationGetNumOperands(terminator))),
            );
            i = 0;
            while (i < bind_count) : (i += 1) {
                const after_arg = mlir.oraBlockGetArgument(owner_block, @intCast(i));
                const yielded_value = mlir.oraOperationGetOperand(terminator, @intCast(i));
                try self.encoder.bindValue(after_arg, try self.encoder.encodeValue(yielded_value));
            }

            const encoded = blk: {
                if (try self.encoder.tryEncodeAssertCondition(source_op, .Current)) |specialized| {
                    break :blk specialized;
                }
                break :blk try self.encoder.encodeValue(condition_value);
            };

            var extra_constraint_list = ManagedArrayList(z3.Z3_ast).init(self.allocator);
            defer extra_constraint_list.deinit();
            const raw_extra_constraints = try self.encoder.takeConstraints(self.allocator);
            defer if (raw_extra_constraints.len > 0) self.allocator.free(raw_extra_constraints);
            try addConstraintSlice(&extra_constraint_list, raw_extra_constraints);
            try addConstraintSlice(&extra_constraint_list, head_constraint_list.items);
            try extra_constraint_list.append(head_condition);
            try addExplicitOldLinkConstraintsForAst(self, &extra_constraint_list, encoded);

            if (ann.loop_step_body_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_step_body_extra_constraints);
            }
            if (ann.loop_step_head_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_step_head_extra_constraints);
            }
            ann.loop_step_head_condition = head_condition;
            ann.loop_step_head_extra_constraints = try head_constraint_list.toOwnedSlice();
            ann.loop_step_body_condition = encoded;
            ann.loop_step_body_extra_constraints = try extra_constraint_list.toOwnedSlice();
        }

        try self.restoreEncoderBranchState(&base_encoder_state);
    }

    fn encodeLoopContinueCondition(self: *VerificationPass, invariant_op: mlir.MlirOperation) !?z3.Z3_ast {
        const loop_op = self.findEnclosingLoopOp(invariant_op) orelse return null;
        const loop_name_ref = mlir.oraOperationGetName(loop_op);
        defer @import("mlir_c_api").freeStringRef(loop_name_ref);
        const loop_name = if (loop_name_ref.data == null or loop_name_ref.length == 0)
            ""
        else
            loop_name_ref.data[0..loop_name_ref.length];

        if (std.mem.eql(u8, loop_name, "scf.while")) {
            const before_block = mlir.oraScfWhileOpGetBeforeBlock(loop_op);
            if (mlir.oraBlockIsNull(before_block)) return null;
            const after_block = mlir.oraScfWhileOpGetAfterBlock(loop_op);

            var rebound_before_args: usize = 0;
            if (!mlir.oraBlockIsNull(after_block) and mlir.mlirOperationGetBlock(invariant_op).ptr == after_block.ptr) {
                const bind_count = @min(
                    @as(usize, @intCast(mlir.oraBlockGetNumArguments(before_block))),
                    @as(usize, @intCast(mlir.oraBlockGetNumArguments(after_block))),
                );
                var i: usize = 0;
                while (i < bind_count) : (i += 1) {
                    const before_arg = mlir.oraBlockGetArgument(before_block, @intCast(i));
                    const after_arg = mlir.oraBlockGetArgument(after_block, @intCast(i));
                    try self.encoder.bindValue(before_arg, try self.encoder.encodeValue(after_arg));
                    rebound_before_args += 1;
                }
            }
            defer {
                var i: usize = 0;
                while (i < rebound_before_args) : (i += 1) {
                    const before_arg = mlir.oraBlockGetArgument(before_block, @intCast(i));
                    _ = self.encoder.value_bindings.remove(@intFromPtr(before_arg.ptr));
                }
            }

            var op = mlir.oraBlockGetFirstOperation(before_block);
            while (!mlir.oraOperationIsNull(op)) {
                const op_name_ref = mlir.oraOperationGetName(op);
                defer @import("mlir_c_api").freeStringRef(op_name_ref);
                const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
                    ""
                else
                    op_name_ref.data[0..op_name_ref.length];

                if (std.mem.eql(u8, op_name, "scf.condition")) {
                    const num_operands = mlir.oraOperationGetNumOperands(op);
                    if (num_operands < 1) return null;
                    const continue_value = mlir.oraOperationGetOperand(op, 0);
                    return try self.encoder.encodeValue(continue_value);
                }
                op = mlir.oraOperationGetNextInBlock(op);
            }
            return null;
        }

        if (std.mem.eql(u8, loop_name, "scf.for")) {
            const body_block = mlir.oraScfForOpGetBodyBlock(loop_op);
            if (mlir.oraBlockIsNull(body_block)) return null;

            const num_block_args = mlir.oraBlockGetNumArguments(body_block);
            if (num_block_args < 1) return null;
            const induction_var = mlir.oraBlockGetArgument(body_block, 0);

            const num_operands = mlir.oraOperationGetNumOperands(loop_op);
            if (num_operands < 2) return null;
            const upper_bound = mlir.oraOperationGetOperand(loop_op, 1);

            const iv_ast = try self.encoder.encodeValue(induction_var);
            const ub_ast = try self.encoder.encodeValue(upper_bound);

            const unsigned_cmp = mlir_helpers.getScfForUnsignedCmp(loop_op);

            return self.buildNumericLt(iv_ast, ub_ast, unsigned_cmp);
        }

        return null;
    }

    fn encodeLoopExitCondition(self: *VerificationPass, invariant_op: mlir.MlirOperation) !?z3.Z3_ast {
        const continue_condition = try self.encodeLoopContinueCondition(invariant_op) orelse return null;
        return z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(continue_condition));
    }

    fn findEnclosingLoopOp(self: *VerificationPass, op: mlir.MlirOperation) ?mlir.MlirOperation {
        _ = self;
        var current_block = mlir.mlirOperationGetBlock(op);
        while (!mlir.oraBlockIsNull(current_block)) {
            const parent_op = mlir.mlirBlockGetParentOperation(current_block);
            if (mlir.oraOperationIsNull(parent_op)) return null;
            const parent_name_ref = mlir.oraOperationGetName(parent_op);
            defer @import("mlir_c_api").freeStringRef(parent_name_ref);
            const parent_name = if (parent_name_ref.data == null or parent_name_ref.length == 0)
                ""
            else
                parent_name_ref.data[0..parent_name_ref.length];

            if (std.mem.eql(u8, parent_name, "scf.while") or std.mem.eql(u8, parent_name, "scf.for")) {
                return parent_op;
            }
            current_block = mlir.mlirOperationGetBlock(parent_op);
        }
        return null;
    }

    fn buildNumericGe(self: *VerificationPass, lhs: z3.Z3_ast, rhs: z3.Z3_ast, unsigned_cmp: bool) ?z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const sort_kind = z3.Z3_get_sort_kind(self.context.ctx, sort);
        if (sort_kind == z3.Z3_BV_SORT) {
            return if (unsigned_cmp)
                z3.Z3_mk_bvuge(self.context.ctx, lhs, rhs)
            else
                z3.Z3_mk_bvsge(self.context.ctx, lhs, rhs);
        }
        return null;
    }

    fn buildNumericLt(self: *VerificationPass, lhs: z3.Z3_ast, rhs: z3.Z3_ast, unsigned_cmp: bool) ?z3.Z3_ast {
        const sort = z3.Z3_get_sort(self.context.ctx, lhs);
        const sort_kind = z3.Z3_get_sort_kind(self.context.ctx, sort);
        if (sort_kind == z3.Z3_BV_SORT) {
            return if (unsigned_cmp)
                z3.Z3_mk_bvult(self.context.ctx, lhs, rhs)
            else
                z3.Z3_mk_bvslt(self.context.ctx, lhs, rhs);
        }
        return null;
    }

    fn logSolverState(self: *VerificationPass, label: []const u8) void {
        if (!self.debug_z3) return;
        const raw = z3.Z3_solver_to_string(self.context.ctx, self.solver.solver);
        if (raw == null) return;
        const c_str: [*:0]const u8 = @ptrCast(raw);
        std.debug.print("[Z3] {s}:\n{s}\n", .{ label, std.mem.span(c_str) });
    }

    fn logAst(self: *VerificationPass, label: []const u8, term: z3.Z3_ast) void {
        if (!self.debug_z3) return;
        const raw = z3.Z3_ast_to_string(self.context.ctx, term);
        if (raw == null) return;
        const c_str: [*:0]const u8 = @ptrCast(raw);
        std.debug.print("[Z3] {s}: {s}\n", .{ label, std.mem.span(c_str) });
    }

    fn traceSmt(self: *const VerificationPass, comptime format: []const u8, args: anytype) void {
        if (!self.trace_smt) return;
        std.debug.print("smt-trace: " ++ format ++ "\n", args);
    }

    fn tracePreparedQuery(self: *const VerificationPass, query_index: usize, query: PreparedQuery) void {
        if (!self.trace_smt) return;
        self.traceSmt(
            "Q{d} kind={s} fn={s} loc={s}:{d}:{d} constraints={d} smt_bytes={d} smt_hash=0x{x}",
            .{
                query_index + 1,
                queryKindLabel(query.kind),
                query.function_name,
                query.file,
                query.line,
                query.column,
                query.constraint_count,
                query.smtlib_bytes,
                query.smtlib_hash,
            },
        );
    }

    fn traceCurrentSolverState(self: *const VerificationPass, phase_label: []const u8) void {
        if (!self.trace_smt) return;
        const raw = z3.Z3_solver_to_string(self.context.ctx, self.solver.solver);
        if (raw == null) {
            self.traceSmt("{s} smt unavailable", .{phase_label});
            return;
        }
        const c_str: [*:0]const u8 = @ptrCast(raw);
        const smt_text = std.mem.span(c_str);
        const smt_hash = std.hash.Wyhash.hash(0, smt_text);
        self.traceSmt(
            "{s} smt_bytes={d} smt_hash=0x{x}",
            .{ phase_label, smt_text.len, smt_hash },
        );
        if (self.trace_smtlib) {
            std.debug.print("smt-trace: {s} smtlib-begin\n{s}\n", .{ phase_label, smt_text });
            std.debug.print("smt-trace: {s} smtlib-end\n", .{phase_label});
        }
    }

    fn phaseLog(self: *const VerificationPass, comptime fmt: []const u8, args: anytype) void {
        if (!self.phase_debug) return;
        std.debug.print("smt-phase: " ++ fmt ++ "\n", args);
    }

    pub fn runVerificationPass(self: *VerificationPass, mlir_module: mlir.MlirModule) !errors.VerificationResult {
        // Transitional routing: use the prepared-query engine as the default
        // execution path so verification semantics come from one query builder.
        // Keep the legacy direct solver path below as an emergency fallback
        // until it can be deleted after parity is confirmed.
        const use_prepared_engine = self.parallel or self.verify_calls or !self.verify_calls;
        if (use_prepared_engine) {
            if (self.parallel) {
                if (self.explain_cores) {
                    self.phaseLog("runVerificationPass -> prepared-sequential (explain mode)", .{});
                    return try self.runVerificationPassPreparedSequential(mlir_module);
                }
                self.phaseLog("runVerificationPass -> prepared-parallel", .{});
                return try self.runVerificationPassParallel(mlir_module);
            }
            self.phaseLog("runVerificationPass -> prepared-sequential", .{});
            return try self.runVerificationPassPreparedSequential(mlir_module);
        }

        self.encoder.clearDegradation();
        self.phaseLog("extract-annotations begin", .{});
        self.extractAnnotationsFromMLIR(mlir_module) catch |err| {
            return try self.annotationExtractionFailureResult(err);
        };
        self.phaseLog("extract-annotations done annotations={d}", .{self.encoded_annotations.items.len});
        if (self.encoder.isDegraded()) {
            self.phaseLog("extract-annotations degraded reason={s}", .{self.encoder.degradationReason() orelse "unknown"});
            return try self.degradedVerificationResult();
        }
        var result = errors.VerificationResult.init(self.allocator);
        var stats_total_queries: u64 = 0;
        var stats_sat: u64 = 0;
        var stats_unsat: u64 = 0;
        var stats_unknown: u64 = 0;
        var stats_total_ms: u64 = 0;

        var by_function = std.StringHashMap(ManagedArrayList(EncodedAnnotation)).init(self.allocator);
        defer {
            var it = by_function.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            by_function.deinit();
        }

        var global_contract_invariants = ManagedArrayList(EncodedAnnotation).init(self.allocator);
        defer global_contract_invariants.deinit();

        for (self.encoded_annotations.items) |ann| {
            if (isGlobalContractInvariantAnnotation(ann)) {
                try global_contract_invariants.append(ann);
                continue;
            }
            const entry = try by_function.getOrPut(ann.function_name);
            if (!entry.found_existing) {
                entry.value_ptr.* = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            }
            try entry.value_ptr.append(ann);
        }
        stableSortAnnotations(global_contract_invariants.items);

        {
            var sort_it = by_function.iterator();
            while (sort_it.next()) |entry| {
                stableSortAnnotations(entry.value_ptr.items);
            }
        }
        stableSortAnnotations(global_contract_invariants.items);

        {
            var sort_it = by_function.iterator();
            while (sort_it.next()) |entry| {
                stableSortAnnotations(entry.value_ptr.items);
            }
        }

        if (global_contract_invariants.items.len > 0) {
            var fn_it = self.encoder.function_ops.iterator();
            while (fn_it.next()) |entry| {
                const fn_name = entry.key_ptr.*;
                const func_op = entry.value_ptr.*;
                if (!self.shouldVerifyFunctionOp(func_op)) continue;
                if (self.filter_function_name) |target_fn| {
                    if (!std.mem.eql(u8, fn_name, target_fn)) continue;
                }
                const fn_entry = try by_function.getOrPut(fn_name);
                if (!fn_entry.found_existing) {
                    fn_entry.value_ptr.* = ManagedArrayList(EncodedAnnotation).init(self.allocator);
                }
            }
        }

        var function_names = try collectSortedFunctionNames(self.allocator, &by_function);
        defer function_names.deinit(self.allocator);

        for (function_names.items) |fn_name| {
            const annotations = by_function.get(fn_name).?.items;
            if (annotations.len == 0 and global_contract_invariants.items.len == 0) continue;
            self.phaseLog("function {s} begin annotations={d} global_invariants={d}", .{ fn_name, annotations.len, global_contract_invariants.items.len });
            var timer = try std.time.Timer.start();

            var assumption_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer assumption_annotations.deinit();
            var path_assumption_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer path_assumption_annotations.deinit();
            var obligation_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer obligation_annotations.deinit();
            var ensure_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer ensure_annotations.deinit();
            var loop_post_invariant_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer loop_post_invariant_annotations.deinit();
            var guard_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer guard_annotations.deinit();

            for (annotations) |ann| {
                if (ann.kind == .RefinementGuard) {
                    try guard_annotations.append(ann);
                } else if (ann.kind == .PathAssume) {
                    try path_assumption_annotations.append(ann);
                } else if (isObligationKind(ann.kind)) {
                    try obligation_annotations.append(ann);
                    if (ann.kind == .Ensures) {
                        try ensure_annotations.append(ann);
                    } else if (ann.kind == .LoopInvariant and ann.loop_exit_condition != null) {
                        try loop_post_invariant_annotations.append(ann);
                    }
                } else if (isAssumptionKind(ann.kind)) {
                    try assumption_annotations.append(ann);
                }
            }

            for (global_contract_invariants.items) |ann| {
                try assumption_annotations.append(ann);
                try obligation_annotations.append(ann);
            }

            var assumption_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
            defer assumption_constraints.deinit();

            try self.solver.resetChecked();
            for (assumption_annotations.items) |ann| {
                for (ann.extra_constraints) |cst| {
                    try assumption_constraints.append(cst);
                }
                try assumption_constraints.append(ann.condition);
            }
            const base_metadata = try buildQueryMetadata(self, assumption_constraints.items);
            const base_tag = try formatQueryTag(self.allocator, base_metadata.constraint_count, base_metadata.smtlib_hash);
            defer self.allocator.free(base_tag);
            for (assumption_constraints.items) |cst| {
                try self.solver.assertChecked(cst);
            }
            if (self.debug_z3) {
                var major: u32 = 0;
                var minor: u32 = 0;
                var build: u32 = 0;
                var rev: u32 = 0;
                z3.Z3_get_version(&major, &minor, &build, &rev);
                std.debug.print("[Z3] version {d}.{d}.{d}.{d}\n", .{ major, minor, build, rev });
                std.debug.print("[Z3] function {s} assumption constraints\n", .{fn_name});
                self.logSolverState("base");
            }

            self.traceSmt("{s} [base] check-start", .{fn_name});
            self.traceCurrentSolverState("query");
            std.debug.print("{s} [base]{s} start\n", .{ fn_name, base_tag });
            timer.reset();
            const assumption_status = try self.solver.checkChecked();
            const elapsed_ms = timer.read() / std.time.ns_per_ms;
            stats_total_queries += 1;
            stats_total_ms += elapsed_ms;
            switch (assumption_status) {
                z3.Z3_L_TRUE => stats_sat += 1,
                z3.Z3_L_FALSE => stats_unsat += 1,
                else => stats_unknown += 1,
            }
            switch (assumption_status) {
                z3.Z3_L_TRUE => {
                    std.debug.print("{s} [base]{s} -> SAT ({d}ms)\n", .{ fn_name, base_tag, elapsed_ms });
                },
                z3.Z3_L_FALSE => {
                    std.debug.print("{s} [base]{s} -> UNSAT ({d}ms)\n", .{ fn_name, base_tag, elapsed_ms });
                    try result.addError(.{
                        .error_type = .PreconditionViolation,
                        .message = try std.fmt.allocPrint(self.allocator, "verification assumptions are inconsistent in {s}", .{fn_name}),
                        .file = try self.allocator.dupe(u8, self.firstLocationFile(annotations)),
                        .line = self.firstLocationLine(annotations),
                        .column = self.firstLocationColumn(annotations),
                        .counterexample = null,
                        .allocator = self.allocator,
                    });
                    return result;
                },
                else => {
                    std.debug.print("{s} [base]{s} -> UNKNOWN ({d}ms)\n", .{ fn_name, base_tag, elapsed_ms });
                    try self.addUnknownVerificationError(
                        &result,
                        try std.fmt.allocPrint(self.allocator, "verification assumptions are unknown in {s}", .{fn_name}),
                        self.firstLocationFile(annotations),
                        self.firstLocationLine(annotations),
                        self.firstLocationColumn(annotations),
                    );
                    continue;
                },
            }

            // Obligation proving: assumptions ∧ ¬obligation must be UNSAT.
            for (obligation_annotations.items) |ann| {
                var relevant_symbols = try buildRelevantSymbolSetForAnnotation(self, ann);
                defer relevant_symbols.deinit();
                var obligation_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer obligation_constraints.deinit();
                try addConstraintSlice(&obligation_constraints, assumption_constraints.items);
                try addApplicablePathAssumptionsToConstraintList(self, &obligation_constraints, path_assumption_annotations.items, ann, &relevant_symbols);
                try addConstraintSlice(&obligation_constraints, ann.path_constraints);
                try addConstraintSlice(&obligation_constraints, ann.extra_constraints);
                try addRelevantOldLinkConstraints(self, &obligation_constraints, &relevant_symbols);
                if (ann.kind == .LoopInvariant) {
                    try addConstraintSlice(&obligation_constraints, ann.loop_entry_extra_constraints);
                }
                if (ann.kind == .ContractInvariant and ann.loop_owner != null) {
                    try addConstraintSlice(&obligation_constraints, ann.loop_step_extra_constraints);
                    for (loop_post_invariant_annotations.items) |peer_inv_ann| {
                        if (!sameLoopInvariantGroup(self, ann, peer_inv_ann)) continue;
                        try addRelevantConstraintSlice(self, &obligation_constraints, peer_inv_ann.path_constraints, &relevant_symbols);
                        try addRelevantConstraintSlice(self, &obligation_constraints, peer_inv_ann.extra_constraints, &relevant_symbols);
                        try obligation_constraints.append(peer_inv_ann.condition);
                    }
                    if (ann.loop_step_condition) |step_cond| {
                        try obligation_constraints.append(step_cond);
                    }
                }
                const negated = z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(ann.condition));
                try obligation_constraints.append(negated);
                const obligation_metadata = try buildQueryMetadata(self, obligation_constraints.items);
                const obligation_tag = try formatQueryTag(self.allocator, obligation_metadata.constraint_count, obligation_metadata.smtlib_hash);
                defer self.allocator.free(obligation_tag);

                try self.solver.pushChecked();
                defer self.solver.pop();
                try addApplicablePathAssumptionsToSolver(self, path_assumption_annotations.items, ann, &relevant_symbols);
                for (ann.path_constraints) |cst| {
                    try self.solver.assertChecked(cst);
                }
                for (ann.extra_constraints) |cst| {
                    try self.solver.assertChecked(cst);
                }
                if (ann.kind == .LoopInvariant) {
                    for (ann.loop_entry_extra_constraints) |cst| {
                        try self.solver.assertChecked(cst);
                    }
                }
                if (ann.kind == .ContractInvariant and ann.loop_owner != null) {
                    for (ann.loop_step_extra_constraints) |cst| {
                        try self.solver.assertChecked(cst);
                    }
                    for (loop_post_invariant_annotations.items) |peer_inv_ann| {
                        if (!sameLoopInvariantGroup(self, ann, peer_inv_ann)) continue;
                        for (peer_inv_ann.path_constraints) |cst| {
                            if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                            try self.solver.assertChecked(cst);
                        }
                        for (peer_inv_ann.extra_constraints) |cst| {
                            if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                            try self.solver.assertChecked(cst);
                        }
                        try self.solver.assertChecked(peer_inv_ann.condition);
                    }
                    if (ann.loop_step_condition) |step_cond| {
                        try self.solver.assertChecked(step_cond);
                    }
                }
                try self.solver.assertChecked(negated);
                if (self.encoder.isDegraded()) {
                    try self.addDegradedDuringProvingError(&result, ann.file, ann.line, ann.column);
                    return result;
                }

                const obligation_label = obligationKindLabel(ann.kind);
                const obligation_log_suffix = try self.annotationLogSuffix(ann);
                defer self.allocator.free(obligation_log_suffix);
                self.traceSmt("{s} [{s}] check-start", .{ fn_name, obligation_label });
                self.traceCurrentSolverState("query");
                std.debug.print("{s} [{s}]{s}{s} start\n", .{ fn_name, obligation_label, obligation_log_suffix, obligation_tag });
                timer.reset();
                const obligation_status = try self.solver.checkChecked();
                const obligation_ms = timer.read() / std.time.ns_per_ms;
                stats_total_queries += 1;
                stats_total_ms += obligation_ms;
                switch (obligation_status) {
                    z3.Z3_L_TRUE => stats_sat += 1,
                    z3.Z3_L_FALSE => stats_unsat += 1,
                    else => stats_unknown += 1,
                }
                std.debug.print("{s} [{s}]{s}{s} -> {s} ({d}ms)\n", .{
                    fn_name,
                    obligation_label,
                    obligation_log_suffix,
                    obligation_tag,
                    switch (obligation_status) {
                        z3.Z3_L_FALSE => "UNSAT",
                        z3.Z3_L_TRUE => "SAT",
                        else => "UNKNOWN",
                    },
                    obligation_ms,
                });

                if (obligation_status == z3.Z3_L_TRUE) {
                    const ce = self.buildCounterexample();
                    try result.addError(.{
                        .error_type = obligationErrorType(ann.kind),
                        .message = try std.fmt.allocPrint(self.allocator, "failed to prove {s} in {s}", .{ obligation_label, fn_name }),
                        .file = try self.allocator.dupe(u8, ann.file),
                        .line = ann.line,
                        .column = ann.column,
                        .counterexample = ce,
                        .allocator = self.allocator,
                    });
                    continue;
                }
                if (obligation_status == z3.Z3_L_UNDEF) {
                    std.debug.print("note: Z3 returned UNKNOWN while proving {s} in {s}.\n", .{ obligation_label, fn_name });
                    try self.addUnknownVerificationError(
                        &result,
                        try std.fmt.allocPrint(self.allocator, "could not prove {s} in {s}", .{ obligation_label, fn_name }),
                        ann.file,
                        ann.line,
                        ann.column,
                    );
                }
                if (obligation_status == z3.Z3_L_FALSE and ann.kind == .Guard and ann.guard_id != null) {
                    const guard_id = ann.guard_id.?;
                    if (!result.proven_guard_ids.contains(guard_id)) {
                        const key = try self.allocator.dupe(u8, guard_id);
                        try result.proven_guard_ids.put(key, {});
                    }
                }

                if (ann.kind == .LoopInvariant) {
                    if (ann.old_condition) |old_inv| {
                        try self.solver.pushChecked();
                        defer self.solver.pop();
                        for (ann.path_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        for (ann.extra_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        for (ann.old_extra_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        for (ann.loop_step_extra_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        for (ann.loop_step_body_extra_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        try self.solver.assertChecked(old_inv);
                        for (loop_post_invariant_annotations.items) |peer_inv_ann| {
                            if (!sameLoopInvariantGroup(self, ann, peer_inv_ann)) continue;
                            for (peer_inv_ann.path_constraints) |cst| {
                                try self.solver.assertChecked(cst);
                            }
                            for (peer_inv_ann.extra_constraints) |cst| {
                                try self.solver.assertChecked(cst);
                            }
                            for (peer_inv_ann.old_extra_constraints) |cst| {
                                try self.solver.assertChecked(cst);
                            }
                            if (peer_inv_ann.old_condition) |peer_old_inv| {
                                try self.solver.assertChecked(peer_old_inv);
                            }
                        }
                        if (ann.loop_step_condition) |step_cond| {
                            try self.solver.assertChecked(step_cond);
                        }
                        const step_body_condition = ann.loop_step_body_condition orelse ann.condition;
                        try self.solver.assertChecked(z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(step_body_condition)));
                        if (self.encoder.isDegraded()) {
                            try self.addDegradedDuringProvingError(&result, ann.file, ann.line, ann.column);
                            return result;
                        }

                        self.traceSmt("{s} [invariant-step] check-start", .{fn_name});
                        self.traceCurrentSolverState("query");
                        std.debug.print("{s} [invariant-step] start\n", .{fn_name});
                        timer.reset();
                        const step_status = try self.solver.checkChecked();
                        const step_ms = timer.read() / std.time.ns_per_ms;
                        stats_total_queries += 1;
                        stats_total_ms += step_ms;
                        switch (step_status) {
                            z3.Z3_L_TRUE => stats_sat += 1,
                            z3.Z3_L_FALSE => stats_unsat += 1,
                            else => stats_unknown += 1,
                        }
                        std.debug.print("{s} [invariant-step] -> {s} ({d}ms)\n", .{
                            fn_name,
                            switch (step_status) {
                                z3.Z3_L_FALSE => "UNSAT",
                                z3.Z3_L_TRUE => "SAT",
                                else => "UNKNOWN",
                            },
                            step_ms,
                        });

                        if (step_status == z3.Z3_L_TRUE) {
                            const ce = self.buildCounterexample();
                            try result.addError(.{
                                .error_type = .InvariantViolation,
                                .message = try std.fmt.allocPrint(self.allocator, "failed to prove loop invariant inductive step in {s}", .{fn_name}),
                                .file = try self.allocator.dupe(u8, ann.file),
                                .line = ann.line,
                                .column = ann.column,
                                .counterexample = ce,
                                .allocator = self.allocator,
                            });
                            continue;
                        }
                        if (step_status == z3.Z3_L_UNDEF) {
                            std.debug.print("note: Z3 returned UNKNOWN while proving invariant step in {s}.\n", .{fn_name});
                            try self.addUnknownVerificationError(
                                &result,
                                try std.fmt.allocPrint(self.allocator, "could not prove loop invariant inductive step in {s}", .{fn_name}),
                                ann.file,
                                ann.line,
                                ann.column,
                            );
                        }
                    }
                }
            }

            // Explicit loop-post proving: assumptions ∧ invariant ∧ exit_condition ∧ ¬ensures must be UNSAT.
            for (loop_post_invariant_annotations.items) |inv_ann| {
                const exit_condition = inv_ann.loop_exit_condition orelse continue;
                for (ensure_annotations.items) |ensure_ann| {
                    try self.solver.pushChecked();
                    defer self.solver.pop();
                    for (ensure_ann.path_constraints) |cst| {
                        try self.solver.assertChecked(cst);
                    }
                    for (ensure_ann.extra_constraints) |cst| {
                        try self.solver.assertChecked(cst);
                    }

                    // Conjoin all invariants for the same loop before discharging
                    // ensures at loop exit. Using a single invariant in isolation
                    // is incomplete and causes false negatives when invariants are
                    // intentionally split across clauses.
                    for (loop_post_invariant_annotations.items) |peer_inv_ann| {
                        if (!sameLoopInvariantGroup(self, inv_ann, peer_inv_ann)) continue;
                        for (peer_inv_ann.path_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        for (peer_inv_ann.extra_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        for (peer_inv_ann.loop_exit_extra_constraints) |cst| {
                            try self.solver.assertChecked(cst);
                        }
                        try self.solver.assertChecked(peer_inv_ann.condition);
                    }

                    try self.solver.assertChecked(exit_condition);
                    try self.solver.assertChecked(z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(ensure_ann.condition)));
                    if (self.encoder.isDegraded()) {
                        try self.addDegradedDuringProvingError(&result, ensure_ann.file, ensure_ann.line, ensure_ann.column);
                        return result;
                    }

                    self.traceSmt("{s} [invariant-post] check-start", .{fn_name});
                    self.traceCurrentSolverState("query");
                    std.debug.print("{s} [invariant-post] start\n", .{fn_name});
                    timer.reset();
                    const post_status = try self.solver.checkChecked();
                    const post_ms = timer.read() / std.time.ns_per_ms;
                    stats_total_queries += 1;
                    stats_total_ms += post_ms;
                    switch (post_status) {
                        z3.Z3_L_TRUE => stats_sat += 1,
                        z3.Z3_L_FALSE => stats_unsat += 1,
                        else => stats_unknown += 1,
                    }
                    std.debug.print("{s} [invariant-post] -> {s} ({d}ms)\n", .{
                        fn_name,
                        switch (post_status) {
                            z3.Z3_L_FALSE => "UNSAT",
                            z3.Z3_L_TRUE => "SAT",
                            else => "UNKNOWN",
                        },
                        post_ms,
                    });

                    if (post_status == z3.Z3_L_TRUE) {
                        const ce = self.buildCounterexample();
                        try result.addError(.{
                            .error_type = .PostconditionViolation,
                            .message = try std.fmt.allocPrint(
                                self.allocator,
                                "failed to prove postcondition from loop invariant at loop exit in {s}",
                                .{fn_name},
                            ),
                            .file = try self.allocator.dupe(u8, ensure_ann.file),
                            .line = ensure_ann.line,
                            .column = ensure_ann.column,
                            .counterexample = ce,
                            .allocator = self.allocator,
                        });
                        continue;
                    }
                    if (post_status == z3.Z3_L_UNDEF) {
                        std.debug.print(
                            "note: Z3 returned UNKNOWN while proving loop-post condition in {s}.\n",
                            .{fn_name},
                        );
                        try self.addUnknownVerificationError(
                            &result,
                            try std.fmt.allocPrint(self.allocator, "could not prove postcondition from loop invariant at loop exit in {s}", .{fn_name}),
                            ensure_ann.file,
                            ensure_ann.line,
                            ensure_ann.column,
                        );
                    }
                }
            }

            // Process refinement guards incrementally:
            // Only guards from path-compatible prefixes are chained as assumptions.
            // Sibling branch guards must not constrain each other.
            var previous_guards = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer previous_guards.deinit();
            for (guard_annotations.items) |ann| {
                if (ann.guard_id == null) continue;
                var relevant_symbols = try buildRelevantSymbolSetForAnnotation(self, ann);
                defer relevant_symbols.deinit();

                // First check: can the guard EVER be satisfied given previous guards?
                // If (assumptions AND previous_guards AND this_guard) is UNSAT, error!
                try self.solver.pushChecked();
                defer self.solver.pop();
                try addApplicablePathAssumptionsToSolver(self, path_assumption_annotations.items, ann, &relevant_symbols);
                for (ann.path_constraints) |cst| {
                    if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                    try self.solver.assertChecked(cst);
                }
                for (previous_guards.items) |prev| {
                    if (!pathConstraintsCompatible(self, prev.path_constraints, ann.path_constraints)) continue;
                    for (prev.path_constraints) |cst| {
                        if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                        try self.solver.assertChecked(cst);
                    }
                    for (prev.extra_constraints) |cst| {
                        if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                        try self.solver.assertChecked(cst);
                    }
                    if (!astUsesOnlyRelevantSymbols(self, prev.condition, &relevant_symbols)) continue;
                    try self.solver.assertChecked(prev.condition);
                }
                for (ann.extra_constraints) |cst| {
                    if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                    try self.solver.assertChecked(cst);
                }
                try self.solver.assertChecked(ann.condition);
                if (self.encoder.isDegraded()) {
                    try self.addDegradedDuringProvingError(&result, ann.file, ann.line, ann.column);
                    return result;
                }
                self.traceSmt("{s} guard {s} [satisfy] check-start", .{ fn_name, ann.guard_id.? });
                self.traceCurrentSolverState("query");
                std.debug.print("{s} guard {s} [satisfy] start\n", .{ fn_name, ann.guard_id.? });
                timer.reset();
                const satisfy_status = try self.solver.checkChecked();
                const satisfy_ms = timer.read() / std.time.ns_per_ms;
                stats_total_queries += 1;
                stats_total_ms += satisfy_ms;
                switch (satisfy_status) {
                    z3.Z3_L_TRUE => stats_sat += 1,
                    z3.Z3_L_FALSE => stats_unsat += 1,
                    else => stats_unknown += 1,
                }
                if (self.debug_z3) {
                    std.debug.print("[Z3] guard satisfiability {s}\n", .{ann.guard_id.?});
                    self.logAst("guard", ann.condition);
                    self.logSolverState("satisfiability");
                }

                if (satisfy_status == z3.Z3_L_FALSE) {
                    // Guard can NEVER be satisfied given previous constraints - error!
                    std.debug.print("{s} guard {s} [satisfy] -> UNSATISFIABLE ({d}ms)\n", .{ fn_name, ann.guard_id.?, satisfy_ms });
                    // No counterexample for UNSAT - there's no satisfying assignment
                    try result.addError(.{
                        .error_type = .RefinementViolation,
                        .message = try std.fmt.allocPrint(self.allocator, "refinement guard can never be satisfied in {s}: {s}", .{ fn_name, ann.guard_id.? }),
                        .file = try self.allocator.dupe(u8, ann.file),
                        .line = ann.line,
                        .column = ann.column,
                        .counterexample = null,
                        .allocator = self.allocator,
                    });
                    return result;
                }
                if (satisfy_status == z3.Z3_L_TRUE) {
                    std.debug.print("{s} guard {s} [satisfy] -> SAT ({d}ms)\n", .{ fn_name, ann.guard_id.?, satisfy_ms });
                } else {
                    std.debug.print("{s} guard {s} [satisfy] -> UNKNOWN ({d}ms)\n", .{ fn_name, ann.guard_id.?, satisfy_ms });
                    try self.addUnknownVerificationError(
                        &result,
                        try std.fmt.allocPrint(self.allocator, "could not prove refinement guard satisfiable in {s}: {s}", .{ fn_name, ann.guard_id.? }),
                        ann.file,
                        ann.line,
                        ann.column,
                    );
                }

                // Second check: can the guard be violated? (to determine if it should be kept)
                try self.solver.pushChecked();
                defer self.solver.pop();
                try addApplicablePathAssumptionsToSolver(self, path_assumption_annotations.items, ann, &relevant_symbols);
                for (ann.path_constraints) |cst| {
                    if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                    try self.solver.assertChecked(cst);
                }
                for (previous_guards.items) |prev| {
                    if (!pathConstraintsCompatible(self, prev.path_constraints, ann.path_constraints)) continue;
                    for (prev.path_constraints) |cst| {
                        if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                        try self.solver.assertChecked(cst);
                    }
                    for (prev.extra_constraints) |cst| {
                        if (!astUsesOnlyRelevantSymbols(self, cst, &relevant_symbols)) continue;
                        try self.solver.assertChecked(cst);
                    }
                    if (!astUsesOnlyRelevantSymbols(self, prev.condition, &relevant_symbols)) continue;
                    try self.solver.assertChecked(prev.condition);
                }
                for (ann.extra_constraints) |cst| {
                    try self.solver.assertChecked(cst);
                }
                const not_guard = z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(ann.condition));
                try self.solver.assertChecked(not_guard);
                if (self.encoder.isDegraded()) {
                    try self.addDegradedDuringProvingError(&result, ann.file, ann.line, ann.column);
                    return result;
                }
                if (self.debug_z3) {
                    std.debug.print("[Z3] guard {s}\n", .{ann.guard_id.?});
                    self.logAst("guard", ann.condition);
                    self.logSolverState("guard");
                }
                self.traceSmt("{s} guard {s} [violate] check-start", .{ fn_name, ann.guard_id.? });
                self.traceCurrentSolverState("query");
                std.debug.print("{s} guard {s} [violate] start\n", .{ fn_name, ann.guard_id.? });
                timer.reset();
                const guard_status = try self.solver.checkChecked();
                const violate_ms = timer.read() / std.time.ns_per_ms;
                stats_total_queries += 1;
                stats_total_ms += violate_ms;
                switch (guard_status) {
                    z3.Z3_L_TRUE => stats_sat += 1,
                    z3.Z3_L_FALSE => stats_unsat += 1,
                    else => stats_unknown += 1,
                }
                if (self.debug_z3) {
                    std.debug.print("[Z3] guard status {s}\n", .{switch (guard_status) {
                        z3.Z3_L_FALSE => "UNSAT",
                        z3.Z3_L_TRUE => "SAT",
                        else => "UNKNOWN",
                    }});
                }
                std.debug.print("{s} guard {s} [violate] -> {s} ({d}ms)\n", .{
                    fn_name,
                    ann.guard_id.?,
                    switch (guard_status) {
                        z3.Z3_L_FALSE => "UNSAT",
                        z3.Z3_L_TRUE => "SAT",
                        else => "UNKNOWN",
                    },
                    violate_ms,
                });
                if (guard_status == z3.Z3_L_FALSE) {
                    const guard_id = ann.guard_id.?;
                    const key = try self.allocator.dupe(u8, guard_id);
                    try result.proven_guard_ids.put(key, {});
                } else if (guard_status == z3.Z3_L_TRUE) {
                    // Guard CAN be violated — extract counterexample before pop
                    if (self.buildCounterexample()) |ce| {
                        try result.diagnostics.append(.{
                            .guard_id = try self.allocator.dupe(u8, ann.guard_id.?),
                            .function_name = try self.allocator.dupe(u8, fn_name),
                            .counterexample = ce,
                            .allocator = self.allocator,
                        });
                    }
                } else {
                    try self.addUnknownVerificationError(
                        &result,
                        try std.fmt.allocPrint(self.allocator, "could not prove refinement guard removable in {s}: {s}", .{ fn_name, ann.guard_id.? }),
                        ann.file,
                        ann.line,
                        ann.column,
                    );
                }

                try previous_guards.append(ann);
            }
            self.phaseLog("function {s} done", .{fn_name});
        }

        if (self.verify_stats) {
            std.debug.print(
                "verification stats: queries={d} sat={d} unsat={d} unknown={d} total_ms={d}\n",
                .{ stats_total_queries, stats_sat, stats_unsat, stats_unknown, stats_total_ms },
            );
        }

        return result;
    }

    pub fn runVerificationPassPreparedSequential(self: *VerificationPass, mlir_module: mlir.MlirModule) !errors.VerificationResult {
        self.encoder.clearDegradation();
        self.phaseLog("prepared-sequential extract-annotations begin", .{});
        self.extractAnnotationsFromMLIR(mlir_module) catch |err| {
            return try self.annotationExtractionFailureResult(err);
        };
        self.phaseLog("prepared-sequential extract-annotations done annotations={d}", .{self.encoded_annotations.items.len});
        if (self.encoder.isDegraded()) {
            self.phaseLog("prepared-sequential degraded after extract", .{});
            return try self.degradedVerificationResult();
        }
        self.phaseLog("prepared-sequential build-queries begin", .{});
        var queries = try self.buildPreparedQueries();
        defer {
            for (queries.items) |*query| {
                query.deinit(self.allocator);
            }
            queries.deinit();
        }
        self.phaseLog("prepared-sequential build-queries done queries={d}", .{queries.items.len});
        if (self.encoder.isDegraded()) {
            self.phaseLog("prepared-sequential degraded after build-queries", .{});
            return try self.degradedVerificationResult();
        }

        if (queries.items.len == 0) {
            return errors.VerificationResult.init(self.allocator);
        }

        const results = try self.allocator.alloc(PreparedQueryResult, queries.items.len);
        defer self.allocator.free(results);
        for (results) |*entry| entry.* = .{};
        defer {
            for (results) |entry| {
                if (entry.model_str) |model| self.allocator.free(model);
                if (entry.explain_str) |explain| self.allocator.free(explain);
                if (entry.explain_tags.len > 0) self.allocator.free(entry.explain_tags);
            }
        }

        for (queries.items, 0..) |query, idx| {
            if (self.trace_smt) {
                self.tracePreparedQuery(idx, query);
                self.traceSmt("Q{d} load-smt begin", .{idx + 1});
                if (self.trace_smtlib) {
                    std.debug.print("smt-trace: Q{d} smtlib-begin\n{s}\n", .{ idx + 1, query.smtlib_z });
                    std.debug.print("smt-trace: Q{d} smtlib-end\n", .{idx + 1});
                }
            }

            try self.solver.resetChecked();
            if (self.timeout_ms) |ms| {
                try self.solver.setTimeoutMs(ms);
            }

            std.debug.print("{s} start\n", .{query.log_prefix});
            var timer = try std.time.Timer.start();
            const status = if (self.explain_cores) blk: {
                try assertPreparedQueryUntrackedConstraints(&self.solver, query);

                if (query.kind != .Base) {
                    const vacuity = try checkPreparedQueryTrackedAssumptions(self, query, false);
                    if (vacuity.status == z3.Z3_L_FALSE) {
                        if (vacuity.explain_str) |core| {
                            std.debug.print("{s} note: assumptions inconsistent ({s})\n", .{ query.log_prefix, core });
                        } else {
                            std.debug.print("{s} note: assumptions inconsistent\n", .{query.log_prefix});
                        }
                        results[idx].vacuous = true;
                        results[idx].explain_str = vacuity.explain_str;
                        results[idx].explain_tags = vacuity.explain_tags;
                        break :blk vacuity.status;
                    }
                    if (vacuity.explain_str) |core| self.allocator.free(core);
                    if (vacuity.explain_tags.len > 0) self.allocator.free(vacuity.explain_tags);
                }

                try self.solver.resetChecked();
                if (self.timeout_ms) |ms| {
                    try self.solver.setTimeoutMs(ms);
                }
                try assertPreparedQueryUntrackedConstraints(&self.solver, query);
                const proof = try checkPreparedQueryTrackedAssumptions(self, query, true);
                if (proof.status == z3.Z3_L_FALSE) {
                    if (proof.explain_str) |core| {
                        std.debug.print("{s} core: {s}\n", .{ query.log_prefix, core });
                    }
                }
                results[idx].explain_str = proof.explain_str;
                results[idx].explain_tags = proof.explain_tags;
                break :blk proof.status;
            } else blk: {
                try assertPreparedQueryConstraints(&self.solver, query.constraints);
                break :blk try self.solver.checkChecked();
            };
            const elapsed_ms = timer.read() / std.time.ns_per_ms;

            std.debug.print("{s} -> {s} ({d}ms)\n", .{
                query.log_prefix,
                queryStatusLabel(status),
                elapsed_ms,
            });
            if (status == z3.Z3_L_UNDEF) {
                std.debug.print("note: Z3 returned UNKNOWN (likely timeout). Consider adding stronger constraints or increasing ORA_Z3_TIMEOUT_MS.\n", .{});
            }

            results[idx].status = status;
            results[idx].elapsed_ms = elapsed_ms;

            if (status == z3.Z3_L_TRUE and (query.kind == .GuardViolate or query.kind == .Obligation or query.kind == .LoopInvariantStep or query.kind == .LoopInvariantPost)) {
                if (try self.solver.getModelChecked()) |model| {
                    const raw = z3.Z3_model_to_string(self.context.ctx, model);
                    if (raw != null) {
                        results[idx].model_str = try self.allocator.dupe(u8, std.mem.span(raw));
                    }
                }
            }
        }

        if (self.verify_stats) {
            var total_queries_stats: u64 = 0;
            var sat: u64 = 0;
            var unsat: u64 = 0;
            var unknown: u64 = 0;
            var total_ms: u64 = 0;
            for (results) |entry| {
                total_queries_stats += 1;
                total_ms += entry.elapsed_ms;
                switch (entry.status) {
                    z3.Z3_L_TRUE => sat += 1,
                    z3.Z3_L_FALSE => unsat += 1,
                    else => unknown += 1,
                }
            }
            std.debug.print(
                "verification stats: queries={d} sat={d} unsat={d} unknown={d} total_ms={d}\n",
                .{ total_queries_stats, sat, unsat, unknown, total_ms },
            );
        }

        return try self.collectPreparedQueryResults(queries.items, results);
    }

    fn runVerificationPassParallel(self: *VerificationPass, mlir_module: mlir.MlirModule) !errors.VerificationResult {
        self.encoder.clearDegradation();
        self.phaseLog("parallel extract-annotations begin", .{});
        self.extractAnnotationsFromMLIR(mlir_module) catch |err| {
            return try self.annotationExtractionFailureResult(err);
        };
        self.phaseLog("parallel extract-annotations done annotations={d}", .{self.encoded_annotations.items.len});
        if (self.encoder.isDegraded()) {
            self.phaseLog("parallel degraded after extract", .{});
            return try self.degradedVerificationResult();
        }
        self.phaseLog("parallel build-queries begin", .{});
        var queries = try self.buildPreparedQueries();
        defer {
            for (queries.items) |*query| {
                query.deinit(self.allocator);
            }
            queries.deinit();
        }
        self.phaseLog("parallel build-queries done queries={d}", .{queries.items.len});
        if (self.encoder.isDegraded()) {
            self.phaseLog("parallel degraded after build-queries", .{});
            return try self.degradedVerificationResult();
        }

        if (queries.items.len == 0) {
            return errors.VerificationResult.init(self.allocator);
        }

        if (self.trace_smt) {
            for (queries.items, 0..) |query, idx| {
                self.tracePreparedQuery(idx, query);
            }
        }

        var thread_safe_allocator = std.heap.ThreadSafeAllocator{ .child_allocator = std.heap.page_allocator };
        const worker_allocator = thread_safe_allocator.allocator();

        const total_queries = queries.items.len;
        const worker_count = @min(self.max_workers, total_queries);

        const results = try worker_allocator.alloc(PreparedQueryResult, total_queries);
        for (results) |*entry| {
            entry.* = .{};
        }
        defer worker_allocator.free(results);

        const WorkState = struct {
            queries: []const PreparedQuery,
            next_index: std.atomic.Value(usize),
            results: []PreparedQueryResult,
            allocator: std.mem.Allocator,
            timeout_ms: ?u32,
            trace_smt: bool,
            trace_smtlib: bool,
            setup_error_mutex: std.Thread.Mutex = .{},
            setup_error: ?anyerror = null,

            fn recordSetupError(state: *@This(), err: anyerror) void {
                state.setup_error_mutex.lock();
                defer state.setup_error_mutex.unlock();
                if (state.setup_error == null) {
                    state.setup_error = err;
                }
            }
        };

        var state = WorkState{
            .queries = queries.items,
            .next_index = std.atomic.Value(usize).init(0),
            .results = results,
            .allocator = worker_allocator,
            .timeout_ms = scaledParallelTimeoutMs(self.timeout_ms, worker_count),
            .trace_smt = self.trace_smt,
            .trace_smtlib = self.trace_smtlib,
        };

        const Worker = struct {
            fn run(ctx: *WorkState) void {
                var context = Context.init(ctx.allocator) catch |err| {
                    ctx.recordSetupError(err);
                    return;
                };
                defer context.deinit();

                var solver = Solver.init(&context, ctx.allocator) catch |err| {
                    ctx.recordSetupError(err);
                    return;
                };
                defer solver.deinit();

                while (true) {
                    const idx = ctx.next_index.fetchAdd(1, .seq_cst);
                    if (idx >= ctx.queries.len) break;

                    const query = ctx.queries[idx];
                    if (ctx.trace_smt) {
                        std.debug.print(
                            "smt-trace: Q{d} reset kind={s} fn={s} constraints={d} smt_bytes={d} smt_hash=0x{x}\n",
                            .{
                                idx + 1,
                                queryKindLabel(query.kind),
                                query.function_name,
                                query.constraint_count,
                                query.smtlib_bytes,
                                query.smtlib_hash,
                            },
                        );
                    }
                    solver.resetChecked() catch |err| {
                        ctx.results[idx].err = err;
                        continue;
                    };
                    if (ctx.trace_smt) {
                        std.debug.print("smt-trace: Q{d} load-smt begin\n", .{idx + 1});
                        if (ctx.trace_smtlib) {
                            std.debug.print("smt-trace: Q{d} smtlib-begin\n{s}\n", .{ idx + 1, query.smtlib_z });
                            std.debug.print("smt-trace: Q{d} smtlib-end\n", .{idx + 1});
                        }
                    }
                    solver.loadFromSmtlib(query.smtlib_z, query.decl_symbols, query.decls) catch |err| {
                        ctx.results[idx].err = err;
                        continue;
                    };
                    if (ctx.timeout_ms) |ms| {
                        solver.setTimeoutMs(ms) catch |err| {
                            ctx.results[idx].err = err;
                            continue;
                        };
                    }
                    if (ctx.trace_smt) {
                        std.debug.print("smt-trace: Q{d} load-smt done\n", .{idx + 1});
                        std.debug.print("smt-trace: Q{d} check start\n", .{idx + 1});
                    }

                    std.debug.print("{s} start\n", .{query.log_prefix});
                    var timer = std.time.Timer.start() catch |err| {
                        ctx.results[idx].err = err;
                        continue;
                    };
                    const status = solver.checkChecked() catch |err| {
                        ctx.results[idx].err = err;
                        continue;
                    };
                    const elapsed_ms = timer.read() / std.time.ns_per_ms;
                    if (ctx.trace_smt) {
                        std.debug.print(
                            "smt-trace: Q{d} check done status={s} elapsed_ms={d}\n",
                            .{
                                idx + 1,
                                switch (status) {
                                    z3.Z3_L_FALSE => "UNSAT",
                                    z3.Z3_L_TRUE => "SAT",
                                    else => "UNKNOWN",
                                },
                                elapsed_ms,
                            },
                        );
                    }

                    std.debug.print("{s} -> {s} ({d}ms)\n", .{
                        query.log_prefix,
                        switch (status) {
                            z3.Z3_L_FALSE => "UNSAT",
                            z3.Z3_L_TRUE => "SAT",
                            else => "UNKNOWN",
                        },
                        elapsed_ms,
                    });
                    if (status == z3.Z3_L_UNDEF) {
                        std.debug.print("note: Z3 returned UNKNOWN (likely timeout). Consider adding stronger constraints or increasing ORA_Z3_TIMEOUT_MS.\n", .{});
                    }

                    ctx.results[idx].status = status;
                    ctx.results[idx].elapsed_ms = elapsed_ms;

                    // Capture model string for SAT queries that surface counterexamples.
                    if (status == z3.Z3_L_TRUE and (query.kind == .GuardViolate or query.kind == .Obligation or query.kind == .LoopInvariantStep or query.kind == .LoopInvariantPost)) {
                        if (solver.getModelChecked() catch |err| {
                            ctx.results[idx].err = err;
                            continue;
                        }) |model| {
                            const raw = z3.Z3_model_to_string(context.ctx, model);
                            if (raw != null) {
                                const dup = ctx.allocator.dupe(u8, std.mem.span(raw)) catch null;
                                ctx.results[idx].model_str = dup;
                                if (ctx.trace_smt and dup != null) {
                                    std.debug.print("smt-trace: Q{d} model captured bytes={d}\n", .{ idx + 1, dup.?.len });
                                }
                            }
                        }
                    }
                }
            }
        };

        var threads = try worker_allocator.alloc(std.Thread, worker_count);
        defer worker_allocator.free(threads);

        var launched: usize = 0;
        while (launched < worker_count) : (launched += 1) {
            threads[launched] = try std.Thread.spawn(.{}, Worker.run, .{&state});
        }
        for (threads[0..launched]) |t| t.join();

        if (state.setup_error) |err| return err;

        const combined = try self.collectPreparedQueryResults(queries.items, results);

        // Free captured model strings
        for (results) |entry| {
            if (entry.model_str) |ms| {
                worker_allocator.free(ms);
            }
        }

        if (self.verify_stats) {
            var total_queries_stats: u64 = 0;
            var sat: u64 = 0;
            var unsat: u64 = 0;
            var unknown: u64 = 0;
            var total_ms: u64 = 0;
            for (results) |entry| {
                total_queries_stats += 1;
                total_ms += entry.elapsed_ms;
                switch (entry.status) {
                    z3.Z3_L_TRUE => sat += 1,
                    z3.Z3_L_FALSE => unsat += 1,
                    else => unknown += 1,
                }
            }
            std.debug.print(
                "verification stats: queries={d} sat={d} unsat={d} unknown={d} total_ms={d}\n",
                .{ total_queries_stats, sat, unsat, unknown, total_ms },
            );
        }

        return combined;
    }

    fn collectPreparedQueryResults(
        self: *VerificationPass,
        queries: []const PreparedQuery,
        results: []const PreparedQueryResult,
    ) !errors.VerificationResult {
        std.debug.assert(queries.len == results.len);

        var combined = errors.VerificationResult.init(self.allocator);
        var inconsistent_functions = std.StringHashMap(void).init(self.allocator);
        defer {
            var fn_it = inconsistent_functions.iterator();
            while (fn_it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            inconsistent_functions.deinit();
        }
        var unknown_base_functions = std.StringHashMap(void).init(self.allocator);
        defer {
            var fn_it = unknown_base_functions.iterator();
            while (fn_it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            unknown_base_functions.deinit();
        }

        for (results, 0..) |entry, idx| {
            if (entry.err) |err| return err;
            const query = queries[idx];
            if (query.kind != .Base) continue;

            if (entry.status == z3.Z3_L_FALSE) {
                if (!inconsistent_functions.contains(query.function_name)) {
                    const fn_name_copy = try self.allocator.dupe(u8, query.function_name);
                    try inconsistent_functions.put(fn_name_copy, {});
                }
                try combined.addError(.{
                    .error_type = .PreconditionViolation,
                    .message = try std.fmt.allocPrint(self.allocator, "verification assumptions are inconsistent in {s}", .{query.function_name}),
                    .file = try self.allocator.dupe(u8, query.file),
                    .line = query.line,
                    .column = query.column,
                    .counterexample = null,
                    .allocator = self.allocator,
                });
            } else if (entry.status == z3.Z3_L_UNDEF) {
                if (!unknown_base_functions.contains(query.function_name)) {
                    const fn_name_copy = try self.allocator.dupe(u8, query.function_name);
                    try unknown_base_functions.put(fn_name_copy, {});
                }
                try self.addUnknownVerificationError(
                    &combined,
                    try std.fmt.allocPrint(self.allocator, "verification assumptions are unknown in {s}", .{query.function_name}),
                    query.file,
                    query.line,
                    query.column,
                );
            }
        }

        for (results, 0..) |entry, idx| {
            const query = queries[idx];
            if (query.kind == .Base) continue;
            if (inconsistent_functions.contains(query.function_name)) continue;
            if (unknown_base_functions.contains(query.function_name)) continue;

            switch (query.kind) {
                .Obligation => {
                    if (entry.status == z3.Z3_L_TRUE) {
                        const ce = if (entry.model_str) |model_str| parseModelString(self.allocator, model_str) else null;
                        try combined.addError(.{
                            .error_type = obligationErrorType(query.obligation_kind orelse .Ensures),
                            .message = try std.fmt.allocPrint(
                                self.allocator,
                                "failed to prove {s} in {s}",
                                .{ obligationKindLabel(query.obligation_kind orelse .Ensures), query.function_name },
                            ),
                            .file = try self.allocator.dupe(u8, query.file),
                            .line = query.line,
                            .column = query.column,
                            .counterexample = ce,
                            .allocator = self.allocator,
                        });
                    } else if (entry.status == z3.Z3_L_UNDEF) {
                        try self.addUnknownVerificationError(
                            &combined,
                            try std.fmt.allocPrint(
                                self.allocator,
                                "could not prove {s} in {s}",
                                .{ obligationKindLabel(query.obligation_kind orelse .Ensures), query.function_name },
                            ),
                            query.file,
                            query.line,
                            query.column,
                        );
                    } else if (entry.status == z3.Z3_L_FALSE and query.obligation_kind == .Guard and query.guard_id != null) {
                        const guard_id = query.guard_id.?;
                        if (!combined.proven_guard_ids.contains(guard_id)) {
                            const key = try self.allocator.dupe(u8, guard_id);
                            try combined.proven_guard_ids.put(key, {});
                        }
                    }
                },
                .LoopInvariantStep => {
                    if (entry.status == z3.Z3_L_TRUE) {
                        const ce = if (entry.model_str) |model_str| parseModelString(self.allocator, model_str) else null;
                        try combined.addError(.{
                            .error_type = .InvariantViolation,
                            .message = try std.fmt.allocPrint(
                                self.allocator,
                                "failed to prove loop invariant inductive step in {s}",
                                .{query.function_name},
                            ),
                            .file = try self.allocator.dupe(u8, query.file),
                            .line = query.line,
                            .column = query.column,
                            .counterexample = ce,
                            .allocator = self.allocator,
                        });
                    } else if (entry.status == z3.Z3_L_UNDEF) {
                        try self.addUnknownVerificationError(
                            &combined,
                            try std.fmt.allocPrint(
                                self.allocator,
                                "could not prove loop invariant inductive step in {s}",
                                .{query.function_name},
                            ),
                            query.file,
                            query.line,
                            query.column,
                        );
                    }
                },
                .LoopInvariantPost => {
                    if (entry.status == z3.Z3_L_TRUE) {
                        const ce = if (entry.model_str) |model_str| parseModelString(self.allocator, model_str) else null;
                        try combined.addError(.{
                            .error_type = .PostconditionViolation,
                            .message = try std.fmt.allocPrint(
                                self.allocator,
                                "failed to prove postcondition from loop invariant at loop exit in {s}",
                                .{query.function_name},
                            ),
                            .file = try self.allocator.dupe(u8, query.file),
                            .line = query.line,
                            .column = query.column,
                            .counterexample = ce,
                            .allocator = self.allocator,
                        });
                    } else if (entry.status == z3.Z3_L_UNDEF) {
                        try self.addUnknownVerificationError(
                            &combined,
                            try std.fmt.allocPrint(
                                self.allocator,
                                "could not prove postcondition from loop invariant at loop exit in {s}",
                                .{query.function_name},
                            ),
                            query.file,
                            query.line,
                            query.column,
                        );
                    }
                },
                .GuardSatisfy => {
                    if (entry.status == z3.Z3_L_FALSE) {
                        try combined.addError(.{
                            .error_type = .RefinementViolation,
                            .message = try std.fmt.allocPrint(self.allocator, "refinement guard can never be satisfied in {s}: {s}", .{ query.function_name, query.guard_id.? }),
                            .file = try self.allocator.dupe(u8, query.file),
                            .line = query.line,
                            .column = query.column,
                            .counterexample = null,
                            .allocator = self.allocator,
                        });
                    } else if (entry.status == z3.Z3_L_UNDEF) {
                        try self.addUnknownVerificationError(
                            &combined,
                            try std.fmt.allocPrint(self.allocator, "could not prove refinement guard satisfiable in {s}: {s}", .{ query.function_name, query.guard_id.? }),
                            query.file,
                            query.line,
                            query.column,
                        );
                    }
                },
                .GuardViolate => {
                    if (entry.status == z3.Z3_L_FALSE) {
                        const guard_id = query.guard_id.?;
                        if (!combined.proven_guard_ids.contains(guard_id)) {
                            const key = try self.allocator.dupe(u8, guard_id);
                            try combined.proven_guard_ids.put(key, {});
                        }
                    } else if (entry.status == z3.Z3_L_TRUE) {
                        if (entry.model_str) |model_str| {
                            if (parseModelString(self.allocator, model_str)) |ce| {
                                try combined.diagnostics.append(.{
                                    .guard_id = try self.allocator.dupe(u8, query.guard_id.?),
                                    .function_name = try self.allocator.dupe(u8, query.function_name),
                                    .counterexample = ce,
                                    .allocator = self.allocator,
                                });
                            }
                        }
                    } else if (entry.status == z3.Z3_L_UNDEF) {
                        try self.addUnknownVerificationError(
                            &combined,
                            try std.fmt.allocPrint(self.allocator, "could not prove refinement guard removable in {s}: {s}", .{ query.function_name, query.guard_id.? }),
                            query.file,
                            query.line,
                            query.column,
                        );
                    }
                },
                .Base => {},
            }
        }

        return combined;
    }

    pub fn buildSmtReport(
        self: *VerificationPass,
        mlir_module: mlir.MlirModule,
        source_file: []const u8,
        verification_result: ?*const errors.VerificationResult,
    ) !SmtReportArtifacts {
        if (self.encoded_annotations.items.len == 0) {
            self.phaseLog("report extract-annotations begin", .{});
            self.extractAnnotationsFromMLIR(mlir_module) catch |err| {
                var result = try self.annotationExtractionFailureResult(err);
                defer result.deinit();
                return try self.buildVerificationFailureSmtReport(source_file, &result);
            };
            self.phaseLog("report extract-annotations done annotations={d}", .{self.encoded_annotations.items.len});
        }
        if (self.encoder.isDegraded()) {
            self.phaseLog("report degraded before query build", .{});
            return try self.buildDegradedSmtReport(source_file, verification_result);
        }

        self.phaseLog("report build-queries begin", .{});
        var queries = try self.buildPreparedQueries();
        defer {
            for (queries.items) |*query| {
                query.deinit(self.allocator);
            }
            queries.deinit();
        }
        self.phaseLog("report build-queries done queries={d}", .{queries.items.len});
        if (self.encoder.isDegraded()) {
            self.phaseLog("report degraded after query build", .{});
            return try self.buildDegradedSmtReport(source_file, verification_result);
        }

        const report_runs = try self.allocator.alloc(ReportQueryRun, queries.items.len);
        defer {
            for (report_runs) |entry| {
                if (entry.model) |model| {
                    self.allocator.free(model);
                }
                if (entry.explain) |explain| {
                    self.allocator.free(explain);
                }
                if (entry.explain_tags.len > 0) {
                    self.allocator.free(entry.explain_tags);
                }
            }
            self.allocator.free(report_runs);
        }

        for (queries.items, 0..) |query, idx| {
            if (self.trace_smt) {
                self.tracePreparedQuery(idx, query);
                self.traceSmt("Q{d} report reset", .{idx + 1});
            }
            try self.solver.resetChecked();
            if (self.trace_smt) {
                self.traceSmt("Q{d} report load-smt begin", .{idx + 1});
                if (self.trace_smtlib) {
                    std.debug.print("smt-trace: Q{d} smtlib-begin\n{s}\n", .{ idx + 1, query.smtlib_z });
                    std.debug.print("smt-trace: Q{d} smtlib-end\n", .{idx + 1});
                }
            }
            if (self.timeout_ms) |ms| {
                try self.solver.setTimeoutMs(ms);
            }

            if (self.trace_smt) {
                self.traceSmt("Q{d} report check-start", .{idx + 1});
            }
            var timer = try std.time.Timer.start();
            var explain_copy: ?[]u8 = null;
            var explain_tags_copy: []const AssumptionTag = &.{};
            var vacuous = false;
            const status = if (self.explain_cores) blk: {
                try assertPreparedQueryUntrackedConstraints(&self.solver, query);

                if (query.kind != .Base) {
                    const vacuity = try checkPreparedQueryTrackedAssumptions(self, query, false);
                    if (vacuity.status == z3.Z3_L_FALSE) {
                        vacuous = true;
                        explain_copy = if (vacuity.explain_str) |core|
                            try self.allocator.dupe(u8, core)
                        else
                            null;
                        explain_tags_copy = try cloneAssumptionTagSlice(self.allocator, vacuity.explain_tags);
                        if (vacuity.explain_str) |core| self.allocator.free(core);
                        if (vacuity.explain_tags.len > 0) self.allocator.free(vacuity.explain_tags);
                        break :blk vacuity.status;
                    }
                    if (vacuity.explain_str) |core| self.allocator.free(core);
                    if (vacuity.explain_tags.len > 0) self.allocator.free(vacuity.explain_tags);
                }

                try self.solver.resetChecked();
                if (self.timeout_ms) |ms| {
                    try self.solver.setTimeoutMs(ms);
                }
                try assertPreparedQueryUntrackedConstraints(&self.solver, query);
                const proof = try checkPreparedQueryTrackedAssumptions(self, query, true);
                explain_copy = if (proof.explain_str) |core|
                    try self.allocator.dupe(u8, core)
                else
                    null;
                explain_tags_copy = try cloneAssumptionTagSlice(self.allocator, proof.explain_tags);
                if (proof.explain_str) |core| self.allocator.free(core);
                if (proof.explain_tags.len > 0) self.allocator.free(proof.explain_tags);
                break :blk proof.status;
            } else blk: {
                try assertPreparedQueryConstraints(&self.solver, query.constraints);
                break :blk try self.solver.checkChecked();
            };
            const elapsed_ms = timer.read() / std.time.ns_per_ms;
            if (self.trace_smt) {
                self.traceSmt(
                    "Q{d} report check-done status={s} elapsed_ms={d}",
                    .{ idx + 1, queryStatusLabel(status), elapsed_ms },
                );
            }

            var model_copy: ?[]u8 = null;
            if (status == z3.Z3_L_TRUE and (query.kind == .Obligation or
                query.kind == .LoopInvariantStep or
                query.kind == .LoopInvariantPost or
                query.kind == .GuardViolate))
            {
                if (try self.solver.getModelChecked()) |model| {
                    const raw = z3.Z3_model_to_string(self.context.ctx, model);
                    if (raw != null) {
                        model_copy = try self.allocator.dupe(u8, std.mem.span(raw));
                    }
                }
            }

            report_runs[idx] = .{
                .status = status,
                .elapsed_ms = elapsed_ms,
                .model = model_copy,
                .explain = explain_copy,
                .explain_tags = explain_tags_copy,
                .vacuous = vacuous,
            };
        }

        var summary = ReportSummary{
            .total_queries = @intCast(queries.items.len),
        };
        summary.encoding_degraded = self.encoder.isDegraded();
        summary.degradation_reason = self.encoder.degradationReason();
        var kind_counts = ReportKindCounts{};

        var proven_guard_ids = std.StringHashMap(void).init(self.allocator);
        defer {
            var it = proven_guard_ids.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            proven_guard_ids.deinit();
        }

        var violatable_guard_ids = std.StringHashMap(void).init(self.allocator);
        defer {
            var it = violatable_guard_ids.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            violatable_guard_ids.deinit();
        }

        for (queries.items, report_runs) |query, run| {
            switch (run.status) {
                z3.Z3_L_TRUE => summary.sat += 1,
                z3.Z3_L_FALSE => summary.unsat += 1,
                else => summary.unknown += 1,
            }
            if (run.vacuous) {
                summary.vacuous += 1;
            }

            switch (query.kind) {
                .Base => {
                    kind_counts.base += 1;
                    if (run.status == z3.Z3_L_FALSE) {
                        summary.inconsistent_bases += 1;
                    }
                },
                .Obligation => {
                    kind_counts.obligation += 1;
                    if (run.status == z3.Z3_L_TRUE) {
                        summary.failed_obligations += 1;
                    }
                },
                .LoopInvariantStep => {
                    kind_counts.loop_invariant_step += 1;
                    if (run.status == z3.Z3_L_TRUE) {
                        summary.failed_obligations += 1;
                    }
                },
                .LoopInvariantPost => {
                    kind_counts.loop_invariant_post += 1;
                    if (run.status == z3.Z3_L_TRUE) {
                        summary.failed_obligations += 1;
                    }
                },
                .GuardSatisfy => {
                    kind_counts.guard_satisfy += 1;
                },
                .GuardViolate => {
                    kind_counts.guard_violate += 1;
                    if (query.guard_id) |guard_id| {
                        if (run.status == z3.Z3_L_FALSE and !proven_guard_ids.contains(guard_id)) {
                            try proven_guard_ids.put(try self.allocator.dupe(u8, guard_id), {});
                        } else if (run.status == z3.Z3_L_TRUE and !violatable_guard_ids.contains(guard_id)) {
                            try violatable_guard_ids.put(try self.allocator.dupe(u8, guard_id), {});
                        }
                    }
                },
            }
        }

        self.phaseLog("report summarize begin", .{});

        if (verification_result) |vr| {
            summary.verification_success = inferReportVerificationSuccess(summary, verification_result, self.encoder.isDegraded());
            summary.verification_errors = @intCast(vr.errors.items.len);
            summary.verification_diagnostics = @intCast(vr.diagnostics.items.len);
            summary.proven_guards = @intCast(vr.proven_guard_ids.count());

            var uniq_diag_guards = std.StringHashMap(void).init(self.allocator);
            defer {
                var it = uniq_diag_guards.iterator();
                while (it.next()) |entry| {
                    self.allocator.free(entry.key_ptr.*);
                }
                uniq_diag_guards.deinit();
            }
            for (vr.diagnostics.items) |diag| {
                if (!uniq_diag_guards.contains(diag.guard_id)) {
                    try uniq_diag_guards.put(try self.allocator.dupe(u8, diag.guard_id), {});
                }
            }
            summary.violatable_guards = @intCast(uniq_diag_guards.count());
        } else {
            summary.verification_success = inferReportVerificationSuccess(summary, null, self.encoder.isDegraded());
            summary.verification_errors = 0;
            summary.verification_diagnostics = 0;
            summary.proven_guards = @intCast(proven_guard_ids.count());
            summary.violatable_guards = @intCast(violatable_guard_ids.count());
        }

        const generated_at_unix = std.time.timestamp();
        const markdown = try self.renderSmtReportMarkdown(
            source_file,
            generated_at_unix,
            queries.items,
            report_runs,
            summary,
            kind_counts,
            verification_result,
        );
        errdefer self.allocator.free(markdown);

        const json = try self.renderSmtReportJson(
            source_file,
            generated_at_unix,
            queries.items,
            report_runs,
            summary,
            kind_counts,
            verification_result,
        );
        return .{
            .markdown = markdown,
            .json = json,
        };
    }

    fn buildVerificationFailureSmtReport(
        self: *VerificationPass,
        source_file: []const u8,
        verification_result: *const errors.VerificationResult,
    ) !SmtReportArtifacts {
        const generated_at_unix = std.time.timestamp();
        const summary = ReportSummary{
            .verification_success = false,
            .verification_errors = @intCast(verification_result.errors.items.len),
            .verification_diagnostics = @intCast(verification_result.diagnostics.items.len),
            .proven_guards = @intCast(verification_result.proven_guard_ids.count()),
            .encoding_degraded = false,
            .degradation_reason = self.encoder.degradationReason(),
        };
        const kind_counts = ReportKindCounts{};

        const markdown = try self.renderSmtReportMarkdown(
            source_file,
            generated_at_unix,
            &.{},
            &.{},
            summary,
            kind_counts,
            verification_result,
        );
        errdefer self.allocator.free(markdown);

        const json = try self.renderSmtReportJson(
            source_file,
            generated_at_unix,
            &.{},
            &.{},
            summary,
            kind_counts,
            verification_result,
        );
        return .{
            .markdown = markdown,
            .json = json,
        };
    }

    fn buildDegradedSmtReport(
        self: *VerificationPass,
        source_file: []const u8,
        verification_result: ?*const errors.VerificationResult,
    ) !SmtReportArtifacts {
        const generated_at_unix = std.time.timestamp();
        const summary = ReportSummary{
            .verification_success = false,
            .verification_errors = if (verification_result) |vr| @intCast(vr.errors.items.len) else 0,
            .verification_diagnostics = if (verification_result) |vr| @intCast(vr.diagnostics.items.len) else 0,
            .proven_guards = if (verification_result) |vr| @intCast(vr.proven_guard_ids.count()) else 0,
            .encoding_degraded = true,
            .degradation_reason = self.encoder.degradationReason(),
        };
        const kind_counts = ReportKindCounts{};

        const markdown = try self.renderSmtReportMarkdown(
            source_file,
            generated_at_unix,
            &.{},
            &.{},
            summary,
            kind_counts,
            verification_result,
        );
        errdefer self.allocator.free(markdown);

        const json = try self.renderSmtReportJson(
            source_file,
            generated_at_unix,
            &.{},
            &.{},
            summary,
            kind_counts,
            verification_result,
        );
        return .{
            .markdown = markdown,
            .json = json,
        };
    }

    fn renderSmtReportMarkdown(
        self: *VerificationPass,
        source_file: []const u8,
        generated_at_unix: i64,
        queries: []const PreparedQuery,
        runs: []const ReportQueryRun,
        summary: ReportSummary,
        kind_counts: ReportKindCounts,
        verification_result: ?*const errors.VerificationResult,
    ) ![]u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        const writer = buffer.writer(self.allocator);

        try writer.writeAll("# SMT Encoding Report\n\n");

        try writer.writeAll("## 1. Run Metadata\n");
        try writer.print("- Source file: `{s}`\n", .{source_file});
        try writer.print("- Generated at (unix): `{d}`\n", .{generated_at_unix});
        try writer.print("- Verification mode: `{s}`\n", .{verifyModeLabel(self.verify_mode)});
        try writer.print("- verify_calls: `{any}`\n", .{self.verify_calls});
        try writer.print("- verify_state: `{any}`\n", .{self.verify_state});
        try writer.print("- parallel: `{any}`\n", .{self.parallel});
        try writer.print("- explain_cores: `{any}`\n", .{self.explain_cores});
        try writer.print("- random_seed: `{d}`\n", .{self.random_seed});
        if (self.timeout_ms) |timeout| {
            try writer.print("- timeout_ms: `{d}`\n", .{timeout});
        } else {
            try writer.writeAll("- timeout_ms: `none`\n");
        }
        try writer.writeAll("\n");

        try writer.writeAll("## 2. Summary\n");
        try writer.print("- Total queries: `{d}`\n", .{summary.total_queries});
        try writer.print("- SAT: `{d}`\n", .{summary.sat});
        try writer.print("- UNSAT: `{d}`\n", .{summary.unsat});
        try writer.print("- UNKNOWN: `{d}`\n", .{summary.unknown});
        try writer.print("- Vacuous queries: `{d}`\n", .{summary.vacuous});
        try writer.print("- Failed obligations: `{d}`\n", .{summary.failed_obligations});
        try writer.print("- Inconsistent assumption bases: `{d}`\n", .{summary.inconsistent_bases});
        try writer.print("- Proven guards: `{d}`\n", .{summary.proven_guards});
        try writer.print("- Violatable guards: `{d}`\n", .{summary.violatable_guards});
        try writer.print("- Verification success: `{any}`\n", .{summary.verification_success});
        try writer.print("- Verification errors: `{d}`\n", .{summary.verification_errors});
        try writer.print("- Verification diagnostics: `{d}`\n", .{summary.verification_diagnostics});
        try writer.print("- Encoding degraded: `{any}`\n", .{summary.encoding_degraded});
        if (summary.degradation_reason) |reason| {
            try writer.print("- Degradation reason: `{s}`\n", .{reason});
        }
        try writer.writeAll("\n");

        try writer.writeAll("## 3. Query Kind Counts\n");
        try writer.print("- base: `{d}`\n", .{kind_counts.base});
        try writer.print("- obligation: `{d}`\n", .{kind_counts.obligation});
        try writer.print("- loop_invariant_step: `{d}`\n", .{kind_counts.loop_invariant_step});
        try writer.print("- loop_invariant_post: `{d}`\n", .{kind_counts.loop_invariant_post});
        try writer.print("- guard_satisfy: `{d}`\n", .{kind_counts.guard_satisfy});
        try writer.print("- guard_violate: `{d}`\n", .{kind_counts.guard_violate});
        try writer.writeAll("\n");

        try writer.writeAll("## 4. Findings\n");
        if (verification_result) |vr| {
            if (vr.errors.items.len == 0 and vr.diagnostics.items.len == 0) {
                try writer.writeAll("- No verification findings.\n");
            } else {
                for (vr.errors.items, 0..) |err, idx| {
                    try writer.print("### Error {d}\n", .{idx + 1});
                    try writer.print("- Type: `{s}`\n", .{@tagName(err.error_type)});
                    try writer.print("- Message: {s}\n", .{err.message});
                    try writer.print("- Location: `{s}:{d}:{d}`\n", .{ err.file, err.line, err.column });
                    if (err.counterexample) |ce| {
                        try writer.writeAll("- Counterexample:\n");
                        try writeCounterexampleMarkdown(writer, self.allocator, ce);
                    }
                    try writer.writeAll("\n");
                }

                for (vr.diagnostics.items, 0..) |diag, idx| {
                    try writer.print("### Diagnostic {d}\n", .{idx + 1});
                    try writer.print("- Guard ID: `{s}`\n", .{diag.guard_id});
                    try writer.print("- Function: `{s}`\n", .{diag.function_name});
                    if (diag.counterexample.variables.count() > 0) {
                        try writer.writeAll("- Counterexample:\n");
                    }
                    try writeCounterexampleMarkdown(writer, self.allocator, diag.counterexample);
                    try writer.writeAll("\n");
                }
            }
        } else {
            try writer.writeAll("- Verification findings unavailable (`--no-verify` was used).\n\n");
        }

        try writer.writeAll("## 5. Query Catalog\n");
        for (queries, runs, 0..) |query, run, idx| {
            try writer.print("### Q{d} - {s}\n", .{ idx + 1, queryKindLabel(query.kind) });
            try writer.print("- Function: `{s}`\n", .{query.function_name});
            try writer.print("- Location: `{s}:{d}:{d}`\n", .{ query.file, query.line, query.column });
            try writer.print("- Status: `{s}`\n", .{queryStatusLabel(run.status)});
            try writer.print("- Elapsed ms: `{d}`\n", .{run.elapsed_ms});
            try writer.print("- Constraint count: `{d}`\n", .{query.constraint_count});
            try writer.print("- SMT bytes: `{d}`\n", .{query.smtlib_bytes});
            try writer.print("- SMT hash: `0x{x}`\n", .{query.smtlib_hash});
            try writer.print("- Vacuous: `{any}`\n", .{run.vacuous});
            if (query.guard_id) |guard_id| {
                try writer.print("- Guard ID: `{s}`\n", .{guard_id});
            }
            if (query.obligation_kind) |kind| {
                try writer.print("- Obligation kind: `{s}`\n", .{obligationKindLabel(kind)});
            }
            if (run.explain) |explain| {
                try writer.print("- Explain core: `{s}`\n", .{explain});
            }
            if (run.explain_tags.len > 0) {
                try writer.writeAll("- Explain core tags:\n");
                for (run.explain_tags) |tag| {
                    try writer.print(
                        "  - `{s}` `{s}:{d}:{d}` `{s}`\n",
                        .{ @tagName(tag.kind), tag.file, tag.line, tag.column, tag.label },
                    );
                }
            }
            if (run.model) |model| {
                try writer.writeAll("- Model:\n```smt2\n");
                try writer.writeAll(model);
                try writer.writeAll("\n```\n");
            }
            try writer.writeAll("- SMT-LIB:\n```smt2\n");
            try writer.writeAll(query.smtlib_z);
            try writer.writeAll("\n```\n\n");
        }

        return buffer.toOwnedSlice(self.allocator);
    }

    fn renderSmtReportJson(
        self: *VerificationPass,
        source_file: []const u8,
        generated_at_unix: i64,
        queries: []const PreparedQuery,
        runs: []const ReportQueryRun,
        summary: ReportSummary,
        kind_counts: ReportKindCounts,
        verification_result: ?*const errors.VerificationResult,
    ) ![]u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        const writer = buffer.writer(self.allocator);

        try writer.writeByte('{');
        try writer.writeAll("\"schema\":\"ora.smt.report.v1\",");
        try writer.writeAll("\"source_file\":");
        try writeJsonStringEscaped(writer, source_file);
        try writer.writeByte(',');
        try writer.print("\"generated_at_unix\":{d},", .{generated_at_unix});

        try writer.writeAll("\"settings\":{");
        try writer.writeAll("\"verify_mode\":");
        try writeJsonStringEscaped(writer, verifyModeLabel(self.verify_mode));
        try writer.writeAll(if (self.verify_calls) ",\"verify_calls\":true" else ",\"verify_calls\":false");
        try writer.writeAll(if (self.verify_state) ",\"verify_state\":true" else ",\"verify_state\":false");
        try writer.writeAll(if (self.parallel) ",\"parallel\":true" else ",\"parallel\":false");
        try writer.writeAll(if (self.explain_cores) ",\"explain_cores\":true" else ",\"explain_cores\":false");
        try writer.print(",\"random_seed\":{d}", .{self.random_seed});
        if (self.timeout_ms) |timeout| {
            try writer.print(",\"timeout_ms\":{d}", .{timeout});
        } else {
            try writer.writeAll(",\"timeout_ms\":null");
        }
        try writer.writeByte('}');
        try writer.writeByte(',');

        try writer.writeAll("\"summary\":{");
        try writer.print("\"total_queries\":{d}", .{summary.total_queries});
        try writer.print(",\"sat\":{d}", .{summary.sat});
        try writer.print(",\"unsat\":{d}", .{summary.unsat});
        try writer.print(",\"unknown\":{d}", .{summary.unknown});
        try writer.print(",\"vacuous\":{d}", .{summary.vacuous});
        try writer.print(",\"failed_obligations\":{d}", .{summary.failed_obligations});
        try writer.print(",\"inconsistent_bases\":{d}", .{summary.inconsistent_bases});
        try writer.print(",\"proven_guards\":{d}", .{summary.proven_guards});
        try writer.print(",\"violatable_guards\":{d}", .{summary.violatable_guards});
        try writer.writeAll(if (summary.encoding_degraded) ",\"encoding_degraded\":true" else ",\"encoding_degraded\":false");
        try writer.writeAll(",\"degradation_reason\":");
        if (summary.degradation_reason) |reason| {
            try writeJsonStringEscaped(writer, reason);
        } else {
            try writer.writeAll("null");
        }
        try writer.writeByte('}');
        try writer.writeByte(',');

        try writer.writeAll("\"query_kind_counts\":{");
        try writer.print("\"base\":{d}", .{kind_counts.base});
        try writer.print(",\"obligation\":{d}", .{kind_counts.obligation});
        try writer.print(",\"loop_invariant_step\":{d}", .{kind_counts.loop_invariant_step});
        try writer.print(",\"loop_invariant_post\":{d}", .{kind_counts.loop_invariant_post});
        try writer.print(",\"guard_satisfy\":{d}", .{kind_counts.guard_satisfy});
        try writer.print(",\"guard_violate\":{d}", .{kind_counts.guard_violate});
        try writer.writeByte('}');
        try writer.writeByte(',');

        try writer.writeAll("\"verification\":{");
        try writer.writeAll(if (summary.verification_success) "\"success\":true" else "\"success\":false");
        try writer.print(",\"errors\":{d}", .{summary.verification_errors});
        try writer.print(",\"diagnostics\":{d}", .{summary.verification_diagnostics});
        try writer.writeByte('}');
        try writer.writeByte(',');

        try writer.writeAll("\"errors\":[");
        if (verification_result) |vr| {
            var first_error = true;
            for (vr.errors.items) |err| {
                if (!first_error) try writer.writeByte(',');
                first_error = false;
                const matched_query_idx = findQueryIndexForError(err, queries, runs);
                const classification = if (matched_query_idx) |qidx|
                    classifyQueryFailure(queries[qidx], runs[qidx])
                else
                    FailureClassification{};
                try writer.writeByte('{');
                try writer.writeAll("\"type\":");
                try writeJsonStringEscaped(writer, @tagName(err.error_type));
                try writer.writeAll(",\"message\":");
                try writeJsonStringEscaped(writer, err.message);
                try writer.writeAll(",\"file\":");
                try writeJsonStringEscaped(writer, err.file);
                try writer.print(",\"line\":{d}", .{err.line});
                try writer.print(",\"column\":{d}", .{err.column});
                try writer.writeAll(",\"query_id\":");
                if (matched_query_idx) |qidx| {
                    try writer.print("{d}", .{qidx + 1});
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"subtype\":");
                if (classification.subtype) |subtype| {
                    try writeJsonStringEscaped(writer, subtype);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"confidence\":");
                if (classification.confidence) |confidence| {
                    try writeJsonStringEscaped(writer, confidence);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"evidence\":");
                if (classification.evidence) |evidence| {
                    try writeJsonStringEscaped(writer, evidence);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"narrowing_bits\":");
                if (classification.narrowing_bits) |bits| {
                    try writer.print("{d}", .{bits});
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"refinement_kind\":");
                if (classification.refinement_kind) |refinement_kind| {
                    try writeJsonStringEscaped(writer, refinement_kind);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"counterexample\":");
                if (err.counterexample) |ce| {
                    try writeCounterexampleJson(writer, self.allocator, ce);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeByte('}');
            }
        }
        try writer.writeByte(']');
        try writer.writeByte(',');

        try writer.writeAll("\"diagnostics\":[");
        if (verification_result) |vr| {
            var first_diag = true;
            for (vr.diagnostics.items) |diag| {
                if (!first_diag) try writer.writeByte(',');
                first_diag = false;
                const diag_query_idx = findQueryIndexForDiagnostic(diag, queries, runs);
                const classification = if (diag_query_idx) |qidx|
                    classifyQueryFailure(queries[qidx], runs[qidx])
                else
                    classifyGuardId(diag.guard_id, true);
                try writer.writeByte('{');
                try writer.writeAll("\"guard_id\":");
                try writeJsonStringEscaped(writer, diag.guard_id);
                try writer.writeAll(",\"function_name\":");
                try writeJsonStringEscaped(writer, diag.function_name);
                try writer.writeAll(",\"query_id\":");
                if (diag_query_idx) |qidx| {
                    try writer.print("{d}", .{qidx + 1});
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"subtype\":");
                if (classification.subtype) |subtype| {
                    try writeJsonStringEscaped(writer, subtype);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"confidence\":");
                if (classification.confidence) |confidence| {
                    try writeJsonStringEscaped(writer, confidence);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"evidence\":");
                if (classification.evidence) |evidence| {
                    try writeJsonStringEscaped(writer, evidence);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"refinement_kind\":");
                if (classification.refinement_kind) |refinement_kind| {
                    try writeJsonStringEscaped(writer, refinement_kind);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"counterexample\":");
                try writeCounterexampleJson(writer, self.allocator, diag.counterexample);
                try writer.writeByte('}');
            }
        }
        try writer.writeByte(']');
        try writer.writeByte(',');

        try writer.writeAll("\"queries\":[");
        var first_query = true;
        for (queries, runs, 0..) |query, run, idx| {
            if (!first_query) try writer.writeByte(',');
            first_query = false;
            const classification = classifyQueryFailure(query, run);
            try writer.writeByte('{');
            try writer.print("\"id\":{d}", .{idx + 1});
            try writer.writeAll(",\"kind\":");
            try writeJsonStringEscaped(writer, queryKindLabel(query.kind));
            try writer.writeAll(",\"function_name\":");
            try writeJsonStringEscaped(writer, query.function_name);
            try writer.writeAll(",\"file\":");
            try writeJsonStringEscaped(writer, query.file);
            try writer.print(",\"line\":{d}", .{query.line});
            try writer.print(",\"column\":{d}", .{query.column});
            try writer.writeAll(",\"status\":");
            try writeJsonStringEscaped(writer, queryStatusLabel(run.status));
            try writer.print(",\"elapsed_ms\":{d}", .{run.elapsed_ms});
            try writer.print(",\"constraint_count\":{d}", .{query.constraint_count});
            try writer.print(",\"smtlib_bytes\":{d}", .{query.smtlib_bytes});
            try writer.writeAll(",\"smtlib_hash\":");
            try writer.print("\"0x{x}\"", .{query.smtlib_hash});
            try writer.writeAll(",\"guard_id\":");
            if (query.guard_id) |guard_id| {
                try writeJsonStringEscaped(writer, guard_id);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"obligation_kind\":");
            if (query.obligation_kind) |kind| {
                try writeJsonStringEscaped(writer, obligationKindLabel(kind));
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"subtype\":");
            if (classification.subtype) |subtype| {
                try writeJsonStringEscaped(writer, subtype);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"confidence\":");
            if (classification.confidence) |confidence| {
                try writeJsonStringEscaped(writer, confidence);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"evidence\":");
            if (classification.evidence) |evidence| {
                try writeJsonStringEscaped(writer, evidence);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"narrowing_bits\":");
            if (classification.narrowing_bits) |bits| {
                try writer.print("{d}", .{bits});
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"refinement_kind\":");
            if (classification.refinement_kind) |refinement_kind| {
                try writeJsonStringEscaped(writer, refinement_kind);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"model\":");
            if (run.model) |model| {
                try writeJsonStringEscaped(writer, model);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"vacuous\":");
            try writer.writeAll(if (run.vacuous) "true" else "false");
            try writer.writeAll(",\"explain_core\":");
            if (run.explain) |explain| {
                try writeJsonStringEscaped(writer, explain);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"explain_tags\":[");
            for (run.explain_tags, 0..) |tag, tag_idx| {
                if (tag_idx != 0) try writer.writeByte(',');
                try writeAssumptionTagJson(writer, tag);
            }
            try writer.writeByte(']');
            try writer.writeAll(",\"smtlib\":");
            try writeJsonStringEscaped(writer, query.smtlib_z);
            try writer.writeByte('}');
        }
        try writer.writeByte(']');

        try writer.writeByte('}');
        return buffer.toOwnedSlice(self.allocator);
    }

    fn buildPreparedQueries(self: *VerificationPass) !ManagedArrayList(PreparedQuery) {
        var queries = ManagedArrayList(PreparedQuery).init(self.allocator);
        self.phaseLog("buildPreparedQueries start annotations={d}", .{self.encoded_annotations.items.len});

        var by_function = std.StringHashMap(ManagedArrayList(EncodedAnnotation)).init(self.allocator);
        defer {
            var it = by_function.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            by_function.deinit();
        }

        var global_contract_invariants = ManagedArrayList(EncodedAnnotation).init(self.allocator);
        defer global_contract_invariants.deinit();

        for (self.encoded_annotations.items) |ann| {
            if (isGlobalContractInvariantAnnotation(ann)) {
                try global_contract_invariants.append(ann);
                continue;
            }
            const entry = try by_function.getOrPut(ann.function_name);
            if (!entry.found_existing) {
                entry.value_ptr.* = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            }
            try entry.value_ptr.append(ann);
        }

        if (global_contract_invariants.items.len > 0) {
            var fn_it = self.encoder.function_ops.iterator();
            while (fn_it.next()) |entry| {
                const fn_name = entry.key_ptr.*;
                const func_op = entry.value_ptr.*;
                if (!self.shouldVerifyFunctionOp(func_op)) continue;
                if (self.filter_function_name) |target_fn| {
                    if (!std.mem.eql(u8, fn_name, target_fn)) continue;
                }
                const fn_entry = try by_function.getOrPut(fn_name);
                if (!fn_entry.found_existing) {
                    fn_entry.value_ptr.* = ManagedArrayList(EncodedAnnotation).init(self.allocator);
                }
            }
        }

        var function_names = try collectSortedFunctionNames(self.allocator, &by_function);
        defer function_names.deinit(self.allocator);

        for (function_names.items) |fn_name| {
            const annotations = by_function.get(fn_name).?.items;
            if (annotations.len == 0 and global_contract_invariants.items.len == 0) continue;
            self.phaseLog("buildPreparedQueries function {s} annotations={d}", .{ fn_name, annotations.len });

            var assumption_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer assumption_annotations.deinit();
            var path_assumption_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer path_assumption_annotations.deinit();
            var obligation_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer obligation_annotations.deinit();
            var ensure_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer ensure_annotations.deinit();
            var loop_post_invariant_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer loop_post_invariant_annotations.deinit();
            var guard_annotations = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer guard_annotations.deinit();

            for (annotations) |ann| {
                if (ann.kind == .RefinementGuard) {
                    try guard_annotations.append(ann);
                } else if (ann.kind == .PathAssume) {
                    try path_assumption_annotations.append(ann);
                } else if (isObligationKind(ann.kind)) {
                    try obligation_annotations.append(ann);
                    if (ann.kind == .Ensures) {
                        try ensure_annotations.append(ann);
                    } else if (ann.kind == .LoopInvariant and ann.loop_exit_condition != null) {
                        try loop_post_invariant_annotations.append(ann);
                    }
                } else if (isAssumptionKind(ann.kind)) {
                    try assumption_annotations.append(ann);
                }
            }

            for (global_contract_invariants.items) |ann| {
                try assumption_annotations.append(ann);
                try obligation_annotations.append(ann);
            }

            var assumption_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
            defer assumption_constraints.deinit();
            var tracked_base_assumptions = ManagedArrayList(TrackedAssumption).init(self.allocator);
            defer tracked_base_assumptions.deinit();

            for (assumption_annotations.items) |ann| {
                try addConstraintSlice(&assumption_constraints, ann.extra_constraints);
                try assumption_constraints.append(ann.condition);
            }
            try addTrackedBaseAssumptions(&tracked_base_assumptions, assumption_annotations.items);

            const base_query = try buildSmtlibForConstraints(self.allocator, &self.solver, assumption_constraints.items);
            const base_smtlib = base_query.smtlib_z;
            const base_hash = std.hash.Wyhash.hash(0, base_smtlib);
            const base_tag = try formatQueryTag(self.allocator, assumption_constraints.items.len, base_hash);
            defer self.allocator.free(base_tag);
            const base_log_prefix = try std.fmt.allocPrint(self.allocator, "{s} [base]{s}", .{ fn_name, base_tag });
            try appendPreparedQueryUnique(&queries, self.allocator, .{
                .kind = .Base,
                .function_name = fn_name,
                .file = self.firstLocationFile(annotations),
                .line = self.firstLocationLine(annotations),
                .column = self.firstLocationColumn(annotations),
                .constraints = try cloneConstraintAstSlice(self.allocator, assumption_constraints.items),
                .tracked_assumptions = try cloneTrackedAssumptionSlice(self.allocator, tracked_base_assumptions.items),
                .smtlib_z = base_smtlib,
                .decl_symbols = base_query.decl_symbols,
                .decls = base_query.decls,
                .constraint_count = assumption_constraints.items.len,
                .smtlib_bytes = base_smtlib.len,
                .smtlib_hash = base_hash,
                .log_prefix = base_log_prefix,
            });

            for (obligation_annotations.items) |ann| {
                var relevant_symbols = try buildRelevantSymbolSetForAnnotation(self, ann);
                defer relevant_symbols.deinit();
                var obligation_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer obligation_constraints.deinit();
                var tracked_obligation_assumptions = ManagedArrayList(TrackedAssumption).init(self.allocator);
                defer tracked_obligation_assumptions.deinit();
                try addConstraintSlice(&obligation_constraints, assumption_constraints.items);
                try addTrackedBaseAssumptions(&tracked_obligation_assumptions, assumption_annotations.items);
                try addApplicablePathAssumptionsToConstraintList(self, &obligation_constraints, path_assumption_annotations.items, ann, &relevant_symbols);
                try addApplicableTrackedPathAssumptions(self, &tracked_obligation_assumptions, path_assumption_annotations.items, ann, &relevant_symbols);
                try addConstraintSlice(&obligation_constraints, ann.path_constraints);
                try addConstraintSlice(&obligation_constraints, ann.extra_constraints);
                if (ann.kind == .LoopInvariant) {
                    try addConstraintSlice(&obligation_constraints, ann.loop_entry_extra_constraints);
                }
                if (ann.kind == .ContractInvariant and ann.loop_owner != null) {
                    try addConstraintSlice(&obligation_constraints, ann.loop_step_extra_constraints);
                    for (loop_post_invariant_annotations.items) |peer_inv_ann| {
                        if (!sameLoopInvariantGroup(self, ann, peer_inv_ann)) continue;
                        try addRelevantConstraintSlice(self, &obligation_constraints, peer_inv_ann.path_constraints, &relevant_symbols);
                        try addRelevantConstraintSlice(self, &obligation_constraints, peer_inv_ann.extra_constraints, &relevant_symbols);
                        try obligation_constraints.append(peer_inv_ann.condition);
                        try appendTrackedAssumption(
                            &tracked_obligation_assumptions,
                            peer_inv_ann.condition,
                            makeTrackedAssumptionTag(.loop_invariant, peer_inv_ann),
                        );
                    }
                    if (ann.loop_step_condition) |step_cond| {
                        try obligation_constraints.append(step_cond);
                    }
                }
                const negated = z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(ann.condition));
                try obligation_constraints.append(negated);
                try appendGoalTrackedAssumption(&tracked_obligation_assumptions, fn_name, ann, negated);

                const obligation_query = try buildSmtlibForConstraints(self.allocator, &self.solver, obligation_constraints.items);
                const obligation_smtlib = obligation_query.smtlib_z;
                const obligation_hash = std.hash.Wyhash.hash(0, obligation_smtlib);
                const obligation_tag = try formatQueryTag(self.allocator, obligation_constraints.items.len, obligation_hash);
                defer self.allocator.free(obligation_tag);
                const obligation_log_suffix = try self.annotationLogSuffix(ann);
                defer self.allocator.free(obligation_log_suffix);
                const obligation_log_prefix = try std.fmt.allocPrint(
                    self.allocator,
                    "{s} [{s}]{s}{s}",
                    .{ fn_name, obligationKindLabel(ann.kind), obligation_log_suffix, obligation_tag },
                );
                try appendPreparedQueryUnique(&queries, self.allocator, .{
                    .kind = .Obligation,
                    .function_name = fn_name,
                    .guard_id = ann.guard_id,
                    .obligation_kind = ann.kind,
                    .file = ann.file,
                    .line = ann.line,
                    .column = ann.column,
                    .constraints = try cloneConstraintAstSlice(self.allocator, obligation_constraints.items),
                    .tracked_assumptions = try cloneTrackedAssumptionSlice(self.allocator, tracked_obligation_assumptions.items),
                    .smtlib_z = obligation_smtlib,
                    .decl_symbols = obligation_query.decl_symbols,
                    .decls = obligation_query.decls,
                    .constraint_count = obligation_constraints.items.len,
                    .smtlib_bytes = obligation_smtlib.len,
                    .smtlib_hash = obligation_hash,
                    .log_prefix = obligation_log_prefix,
                });

                if (ann.kind == .LoopInvariant) {
                    if (ann.old_condition) |old_inv| {
                        var step_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                        defer step_constraints.deinit();
                        var tracked_step_assumptions = ManagedArrayList(TrackedAssumption).init(self.allocator);
                        defer tracked_step_assumptions.deinit();
                        try addConstraintSlice(&step_constraints, assumption_constraints.items);
                        try addConstraintSlice(&step_constraints, assumption_constraints.items);
                        try addTrackedBaseAssumptions(&tracked_step_assumptions, assumption_annotations.items);
                        try addConstraintSlice(&step_constraints, ann.path_constraints);
                        if (ann.loop_step_head_condition != null) {
                            try addConstraintSlice(&step_constraints, ann.loop_step_head_extra_constraints);
                            try step_constraints.append(ann.loop_step_head_condition.?);
                            try appendTrackedAssumption(
                                &tracked_step_assumptions,
                                ann.loop_step_head_condition.?,
                                makeTrackedAssumptionTag(.loop_invariant, ann),
                            );
                        } else {
                            try addConstraintSlice(&step_constraints, ann.extra_constraints);
                            try addConstraintSlice(&step_constraints, ann.old_extra_constraints);
                            try step_constraints.append(old_inv);
                            try appendTrackedAssumption(
                                &tracked_step_assumptions,
                                old_inv,
                                makeTrackedAssumptionTag(.loop_invariant, ann),
                            );
                        }
                        try addConstraintSlice(&step_constraints, ann.loop_step_extra_constraints);
                        try addConstraintSlice(&step_constraints, ann.loop_step_body_extra_constraints);
                        for (loop_post_invariant_annotations.items) |peer_inv_ann| {
                            if (!sameLoopInvariantGroup(self, ann, peer_inv_ann)) continue;
                            try addConstraintSlice(&step_constraints, peer_inv_ann.path_constraints);
                            if (peer_inv_ann.loop_step_head_condition != null) {
                                try addConstraintSlice(&step_constraints, peer_inv_ann.loop_step_head_extra_constraints);
                                try step_constraints.append(peer_inv_ann.loop_step_head_condition.?);
                                try appendTrackedAssumption(
                                    &tracked_step_assumptions,
                                    peer_inv_ann.loop_step_head_condition.?,
                                    makeTrackedAssumptionTag(.loop_invariant, peer_inv_ann),
                                );
                            } else {
                                try addConstraintSlice(&step_constraints, peer_inv_ann.extra_constraints);
                                try addConstraintSlice(&step_constraints, peer_inv_ann.old_extra_constraints);
                                if (peer_inv_ann.old_condition) |peer_old_inv| {
                                    try step_constraints.append(peer_old_inv);
                                    try appendTrackedAssumption(
                                        &tracked_step_assumptions,
                                        peer_old_inv,
                                        makeTrackedAssumptionTag(.loop_invariant, peer_inv_ann),
                                    );
                                }
                            }
                        }
                        if (ann.loop_step_condition) |step_cond| {
                            try step_constraints.append(step_cond);
                        }
                        const step_body_condition = ann.loop_step_body_condition orelse ann.condition;
                        const negated_step = z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(step_body_condition));
                        try step_constraints.append(negated_step);
                        try appendGoalTrackedAssumption(&tracked_step_assumptions, fn_name, ann, negated_step);

                        const step_query = try buildSmtlibForConstraints(self.allocator, &self.solver, step_constraints.items);
                        const step_smtlib = step_query.smtlib_z;
                        const step_hash = std.hash.Wyhash.hash(0, step_smtlib);
                        const step_tag = try formatQueryTag(self.allocator, step_constraints.items.len, step_hash);
                        defer self.allocator.free(step_tag);
                        const step_log_prefix = try std.fmt.allocPrint(
                            self.allocator,
                            "{s} [invariant-step]{s}",
                            .{ fn_name, step_tag },
                        );
                        try appendPreparedQueryUnique(&queries, self.allocator, .{
                            .kind = .LoopInvariantStep,
                            .function_name = fn_name,
                            .obligation_kind = .LoopInvariant,
                            .file = ann.file,
                            .line = ann.line,
                            .column = ann.column,
                            .constraints = try cloneConstraintAstSlice(self.allocator, step_constraints.items),
                            .tracked_assumptions = try cloneTrackedAssumptionSlice(self.allocator, tracked_step_assumptions.items),
                            .smtlib_z = step_smtlib,
                            .decl_symbols = step_query.decl_symbols,
                            .decls = step_query.decls,
                            .constraint_count = step_constraints.items.len,
                            .smtlib_bytes = step_smtlib.len,
                            .smtlib_hash = step_hash,
                            .log_prefix = step_log_prefix,
                        });
                    }
                }
            }

            for (loop_post_invariant_annotations.items) |inv_ann| {
                const exit_condition = inv_ann.loop_exit_condition orelse continue;
                for (ensure_annotations.items) |ensure_ann| {
                    var post_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                    defer post_constraints.deinit();
                    var tracked_post_assumptions = ManagedArrayList(TrackedAssumption).init(self.allocator);
                    defer tracked_post_assumptions.deinit();
                    try addConstraintSlice(&post_constraints, assumption_constraints.items);
                    try addTrackedBaseAssumptions(&tracked_post_assumptions, assumption_annotations.items);
                    try addConstraintSlice(&post_constraints, ensure_ann.path_constraints);
                    try addConstraintSlice(&post_constraints, ensure_ann.extra_constraints);

                    // Conjoin all invariants for the same loop in loop-post queries.
                    for (loop_post_invariant_annotations.items) |peer_inv_ann| {
                        if (!sameLoopInvariantGroup(self, inv_ann, peer_inv_ann)) continue;
                        try addConstraintSlice(&post_constraints, peer_inv_ann.path_constraints);
                        try addConstraintSlice(&post_constraints, peer_inv_ann.extra_constraints);
                        try addConstraintSlice(&post_constraints, peer_inv_ann.loop_exit_extra_constraints);
                        try post_constraints.append(peer_inv_ann.condition);
                        try appendTrackedAssumption(
                            &tracked_post_assumptions,
                            peer_inv_ann.condition,
                            makeTrackedAssumptionTag(.loop_invariant, peer_inv_ann),
                        );
                    }

                    try post_constraints.append(exit_condition);
                    const negated_post = z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(ensure_ann.condition));
                    try post_constraints.append(negated_post);
                    try appendGoalTrackedAssumption(&tracked_post_assumptions, fn_name, ensure_ann, negated_post);

                    const post_query = try buildSmtlibForConstraints(self.allocator, &self.solver, post_constraints.items);
                    const post_smtlib = post_query.smtlib_z;
                    const post_hash = std.hash.Wyhash.hash(0, post_smtlib);
                    const post_tag = try formatQueryTag(self.allocator, post_constraints.items.len, post_hash);
                    defer self.allocator.free(post_tag);
                    const post_log_prefix = try std.fmt.allocPrint(
                        self.allocator,
                        "{s} [invariant-post]{s}",
                        .{ fn_name, post_tag },
                    );
                    try appendPreparedQueryUnique(&queries, self.allocator, .{
                        .kind = .LoopInvariantPost,
                        .function_name = fn_name,
                        .obligation_kind = .Ensures,
                        .file = ensure_ann.file,
                        .line = ensure_ann.line,
                        .column = ensure_ann.column,
                        .constraints = try cloneConstraintAstSlice(self.allocator, post_constraints.items),
                        .tracked_assumptions = try cloneTrackedAssumptionSlice(self.allocator, tracked_post_assumptions.items),
                        .smtlib_z = post_smtlib,
                        .decl_symbols = post_query.decl_symbols,
                        .decls = post_query.decls,
                        .constraint_count = post_constraints.items.len,
                        .smtlib_bytes = post_smtlib.len,
                        .smtlib_hash = post_hash,
                        .log_prefix = post_log_prefix,
                    });
                }
            }

            var previous_guards = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer previous_guards.deinit();

            for (guard_annotations.items) |ann| {
                if (ann.guard_id == null) continue;
                var relevant_symbols = try buildRelevantSymbolSetForAnnotation(self, ann);
                defer relevant_symbols.deinit();

                var guard_base = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer guard_base.deinit();
                try addConstraintSlice(&guard_base, assumption_constraints.items);
                try addApplicablePathAssumptionsToConstraintList(self, &guard_base, path_assumption_annotations.items, ann, &relevant_symbols);
                try addRelevantConstraintSlice(self, &guard_base, ann.path_constraints, &relevant_symbols);

                for (previous_guards.items) |prev| {
                    if (!pathConstraintsCompatible(self, prev.path_constraints, ann.path_constraints)) continue;
                    try addRelevantConstraintSlice(self, &guard_base, prev.path_constraints, &relevant_symbols);
                    try addRelevantConstraintSlice(self, &guard_base, prev.extra_constraints, &relevant_symbols);
                    if (!astUsesOnlyRelevantSymbols(self, prev.condition, &relevant_symbols)) continue;
                    try guard_base.append(prev.condition);
                }

                // Satisfy query
                var satisfy_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer satisfy_constraints.deinit();
                try addConstraintSlice(&satisfy_constraints, guard_base.items);
                try addRelevantConstraintSlice(self, &satisfy_constraints, ann.extra_constraints, &relevant_symbols);
                try satisfy_constraints.append(ann.condition);

                const satisfy_query = try buildSmtlibForConstraints(self.allocator, &self.solver, satisfy_constraints.items);
                const satisfy_smtlib = satisfy_query.smtlib_z;
                const satisfy_hash = std.hash.Wyhash.hash(0, satisfy_smtlib);
                const satisfy_tag = try formatQueryTag(self.allocator, satisfy_constraints.items.len, satisfy_hash);
                defer self.allocator.free(satisfy_tag);
                const satisfy_log_prefix = try std.fmt.allocPrint(
                    self.allocator,
                    "{s} guard {s} [satisfy]{s}",
                    .{ fn_name, ann.guard_id.?, satisfy_tag },
                );
                try appendPreparedQueryUnique(&queries, self.allocator, .{
                    .kind = .GuardSatisfy,
                    .function_name = fn_name,
                    .guard_id = ann.guard_id,
                    .file = ann.file,
                    .line = ann.line,
                    .column = ann.column,
                    .constraints = try cloneConstraintAstSlice(self.allocator, satisfy_constraints.items),
                    .smtlib_z = satisfy_smtlib,
                    .decl_symbols = satisfy_query.decl_symbols,
                    .decls = satisfy_query.decls,
                    .constraint_count = satisfy_constraints.items.len,
                    .smtlib_bytes = satisfy_smtlib.len,
                    .smtlib_hash = satisfy_hash,
                    .log_prefix = satisfy_log_prefix,
                });

                // Violate query
                var violate_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer violate_constraints.deinit();
                try addConstraintSlice(&violate_constraints, guard_base.items);
                try addRelevantConstraintSlice(self, &violate_constraints, ann.extra_constraints, &relevant_symbols);
                const not_guard = z3.Z3_mk_not(self.context.ctx, self.encoder.coerceBoolean(ann.condition));
                try violate_constraints.append(not_guard);

                const violate_query = try buildSmtlibForConstraints(self.allocator, &self.solver, violate_constraints.items);
                const violate_smtlib = violate_query.smtlib_z;
                const violate_hash = std.hash.Wyhash.hash(0, violate_smtlib);
                const violate_tag = try formatQueryTag(self.allocator, violate_constraints.items.len, violate_hash);
                defer self.allocator.free(violate_tag);
                const violate_log_prefix = try std.fmt.allocPrint(
                    self.allocator,
                    "{s} guard {s} [violate]{s}",
                    .{ fn_name, ann.guard_id.?, violate_tag },
                );
                try appendPreparedQueryUnique(&queries, self.allocator, .{
                    .kind = .GuardViolate,
                    .function_name = fn_name,
                    .guard_id = ann.guard_id,
                    .file = ann.file,
                    .line = ann.line,
                    .column = ann.column,
                    .constraints = try cloneConstraintAstSlice(self.allocator, violate_constraints.items),
                    .smtlib_z = violate_smtlib,
                    .decl_symbols = violate_query.decl_symbols,
                    .decls = violate_query.decls,
                    .constraint_count = violate_constraints.items.len,
                    .smtlib_bytes = violate_smtlib.len,
                    .smtlib_hash = violate_hash,
                    .log_prefix = violate_log_prefix,
                });

                try previous_guards.append(ann);
            }
            self.phaseLog("buildPreparedQueries function {s} done queries_so_far={d}", .{ fn_name, queries.items.len });
        }

        self.phaseLog("buildPreparedQueries done total_queries={d}", .{queries.items.len});
        return queries;
    }

    fn getLocationInfo(self: *VerificationPass, op: mlir.MlirOperation) !struct { file: []const u8, line: u32, column: u32 } {
        const loc = mlir.oraOperationGetLocation(op);
        if (mlir.oraLocationIsNull(loc)) {
            return .{ .file = "", .line = 0, .column = 0 };
        }

        const loc_ref = mlir.oraLocationPrintToString(loc);
        defer @import("mlir_c_api").freeStringRef(loc_ref);
        if (loc_ref.data == null or loc_ref.length == 0) {
            return .{ .file = "", .line = 0, .column = 0 };
        }
        const loc_str = loc_ref.data[0..loc_ref.length];

        const parsed = parseLocationString(loc_str);
        if (parsed.file.len == 0) {
            return .{ .file = "", .line = 0, .column = 0 };
        }
        const file_copy = try self.allocator.dupe(u8, parsed.file);
        try self.location_storage.append(file_copy);
        return .{ .file = file_copy, .line = parsed.line, .column = parsed.column };
    }

    fn annotationLabelForOp(self: *VerificationPass, op: mlir.MlirOperation, kind: AnnotationKind) !?[]const u8 {
        if (kind == .ContractInvariant or kind == .LoopInvariant) {
            if (try self.getStringAttr(op, "ora.label", &self.label_storage)) |label| return label;
        }
        if (kind == .ContractInvariant or kind == .Guard) {
            if (try self.getStringAttr(op, "message", &self.label_storage)) |message| return message;
        }
        return null;
    }

    fn annotationLogSuffix(self: *VerificationPass, ann: EncodedAnnotation) ![]const u8 {
        const file_name = if (ann.file.len > 0) std.fs.path.basename(ann.file) else "";
        const label = ann.label orelse (try self.inferAnnotationLabelFromSource(ann) orelse "");
        if (file_name.len == 0) {
            return if (label.len > 0)
                try std.fmt.allocPrint(self.allocator, " {s}", .{label})
            else
                try self.allocator.dupe(u8, "");
        }
        if (ann.line > 0 and label.len > 0) {
            return try std.fmt.allocPrint(self.allocator, " {s}:{d} {s}", .{ file_name, ann.line, label });
        }
        if (ann.line > 0) {
            return try std.fmt.allocPrint(self.allocator, " {s}:{d}", .{ file_name, ann.line });
        }
        if (label.len > 0) {
            return try std.fmt.allocPrint(self.allocator, " {s} {s}", .{ file_name, label });
        }
        return try std.fmt.allocPrint(self.allocator, " {s}", .{file_name});
    }

    fn inferAnnotationLabelFromSource(self: *VerificationPass, ann: EncodedAnnotation) !?[]const u8 {
        if (ann.file.len == 0 or ann.line == 0) return null;
        if (ann.kind != .ContractInvariant and ann.kind != .LoopInvariant) return null;
        if (std.mem.startsWith(u8, ann.file, "embedded://")) return null;

        const source_text = std.fs.cwd().readFileAlloc(self.allocator, ann.file, 1 << 20) catch return null;
        defer self.allocator.free(source_text);

        var current_line: u32 = 1;
        var line_start: usize = 0;
        var idx: usize = 0;
        while (idx <= source_text.len) : (idx += 1) {
            const is_end = idx == source_text.len or source_text[idx] == '\n';
            if (!is_end) continue;
            if (current_line == ann.line) {
                const raw_line = std.mem.trim(u8, source_text[line_start..idx], " \t\r");
                if (inferInvariantLabelFromLine(raw_line)) |label| {
                    const dup = try self.allocator.dupe(u8, label);
                    try self.label_storage.append(dup);
                    return dup;
                }
                return null;
            }
            current_line += 1;
            line_start = idx + 1;
        }
        return null;
    }

    fn firstLocationFile(self: *const VerificationPass, annotations: []const EncodedAnnotation) []const u8 {
        _ = self;
        return if (annotations.len > 0) annotations[0].file else "";
    }

    fn firstLocationLine(self: *const VerificationPass, annotations: []const EncodedAnnotation) u32 {
        _ = self;
        return if (annotations.len > 0) annotations[0].line else 0;
    }

    fn firstLocationColumn(self: *const VerificationPass, annotations: []const EncodedAnnotation) u32 {
        _ = self;
        return if (annotations.len > 0) annotations[0].column else 0;
    }

    fn degradedVerificationResult(self: *VerificationPass) !errors.VerificationResult {
        var result = errors.VerificationResult.init(self.allocator);
        const reason = self.encoder.degradationReason() orelse "unknown SMT encoding degradation";
        try self.addDegradedVerificationError(
            &result,
            try std.fmt.allocPrint(
                self.allocator,
                "verification aborted: SMT encoding degraded ({s})",
                .{reason},
            ),
            "",
            0,
            0,
        );
        return result;
    }

    fn addDegradedDuringProvingError(
        self: *VerificationPass,
        result: *errors.VerificationResult,
        file: []const u8,
        line: u32,
        column: u32,
    ) !void {
        const reason = self.encoder.degradationReason() orelse "unknown SMT encoding degradation";
        try self.addDegradedVerificationError(
            result,
            try std.fmt.allocPrint(
                self.allocator,
                "verification aborted: SMT encoding degraded during proving ({s})",
                .{reason},
            ),
            file,
            line,
            column,
        );
    }

    fn addDegradedVerificationError(
        self: *VerificationPass,
        result: *errors.VerificationResult,
        message: []const u8,
        file: []const u8,
        line: u32,
        column: u32,
    ) !void {
        try result.addError(.{
            .error_type = .EncodingDegraded,
            .message = message,
            .file = try self.allocator.dupe(u8, file),
            .line = line,
            .column = column,
            .counterexample = null,
            .allocator = self.allocator,
        });
    }

    fn addUnknownVerificationError(
        self: *VerificationPass,
        result: *errors.VerificationResult,
        message: []const u8,
        file: []const u8,
        line: u32,
        column: u32,
    ) !void {
        try result.addError(.{
            .error_type = .Unknown,
            .message = message,
            .file = try self.allocator.dupe(u8, file),
            .line = line,
            .column = column,
            .counterexample = null,
            .allocator = self.allocator,
        });
    }

    fn annotationExtractionFailureResult(self: *VerificationPass, err: anyerror) !errors.VerificationResult {
        var result = errors.VerificationResult.init(self.allocator);
        const encoder_reason = self.encoder.degradationReason();
        const message = if (encoder_reason) |reason|
            try std.fmt.allocPrint(
                self.allocator,
                "verification aborted during annotation extraction: {s} ({s})",
                .{ @errorName(err), reason },
            )
        else
            try std.fmt.allocPrint(
                self.allocator,
                "verification aborted during annotation extraction: {s}",
                .{@errorName(err)},
            );
        try self.addUnknownVerificationError(&result, message, "", 0, 0);
        return result;
    }

    fn buildCounterexample(self: *VerificationPass) ?errors.Counterexample {
        const model = (self.solver.getModelChecked() catch return null) orelse return null;
        const num_consts = z3.Z3_model_get_num_consts(self.context.ctx, model);
        if (num_consts == 0) return null;

        var ce = errors.Counterexample.init(self.allocator);

        var i: c_uint = 0;
        while (i < num_consts) : (i += 1) {
            const decl = z3.Z3_model_get_const_decl(self.context.ctx, model, i);
            const name_sym = z3.Z3_get_decl_name(self.context.ctx, decl);
            const name_ptr = z3.Z3_get_symbol_string(self.context.ctx, name_sym);
            if (name_ptr == null) continue;
            // Copy name immediately — Z3 reuses internal string buffers
            const name = self.allocator.dupe(u8, std.mem.span(name_ptr)) catch continue;
            defer self.allocator.free(name);

            if (shouldHideCounterexampleVariable(name)) continue;

            const interp = z3.Z3_model_get_const_interp(self.context.ctx, model, decl);
            if (interp == null) continue;
            const val_ptr = z3.Z3_ast_to_string(self.context.ctx, interp);
            if (val_ptr == null) continue;
            // Copy value immediately — same reason
            const val = self.allocator.dupe(u8, std.mem.span(val_ptr)) catch continue;
            defer self.allocator.free(val);

            ce.addVariable(name, val) catch continue;
        }

        if (ce.variables.count() == 0) {
            ce.deinit();
            return null;
        }
        return ce;
    }
};

/// Parse a Z3 model string (from parallel worker) into a Counterexample.
/// Format: "name -> value\n..."
fn parseModelString(allocator: std.mem.Allocator, model_str: []const u8) ?errors.Counterexample {
    var ce = errors.Counterexample.init(allocator);
    const arrow = " -> ";

    var iter = std.mem.splitScalar(u8, model_str, '\n');
    while (iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        const arrow_pos = std.mem.indexOf(u8, trimmed, arrow) orelse continue;
        const name = trimmed[0..arrow_pos];
        const value = trimmed[arrow_pos + arrow.len ..];

        if (shouldHideCounterexampleVariable(name)) continue;

        if (value.len == 0) continue;
        ce.addVariable(name, value) catch continue;
    }

    if (ce.variables.count() == 0) {
        ce.deinit();
        return null;
    }
    return ce;
}

fn astSimplifiesToBool(ctx: z3.Z3_context, ast: z3.Z3_ast) ?bool {
    const simplified = z3.Z3_simplify(ctx, ast);
    if (z3.Z3_get_ast_kind(ctx, simplified) != z3.Z3_APP_AST) return null;
    const app = z3.Z3_to_app(ctx, simplified);
    const decl = z3.Z3_get_app_decl(ctx, app);
    return switch (z3.Z3_get_decl_kind(ctx, decl)) {
        z3.Z3_OP_TRUE => true,
        z3.Z3_OP_FALSE => false,
        else => null,
    };
}

fn shouldSkipHeavyPathConstraint(ctx: z3.Z3_context, ast: z3.Z3_ast) bool {
    if (z3.Z3_get_ast_kind(ctx, ast) != z3.Z3_APP_AST) return false;
    const app = z3.Z3_to_app(ctx, ast);
    const decl = z3.Z3_get_app_decl(ctx, app);
    return switch (z3.Z3_get_decl_kind(ctx, decl)) {
        z3.c.Z3_OP_EQ,
        z3.c.Z3_OP_NOT,
        z3.c.Z3_OP_AND,
        z3.c.Z3_OP_OR,
        z3.c.Z3_OP_XOR,
        z3.c.Z3_OP_ITE,
        => containsHeavyPathArithmetic(ctx, ast),
        else => false,
    };
}

fn containsHeavyPathArithmetic(ctx: z3.Z3_context, ast: z3.Z3_ast) bool {
    if (z3.Z3_get_ast_kind(ctx, ast) != z3.Z3_APP_AST) return false;
    const app = z3.Z3_to_app(ctx, ast);
    const decl = z3.Z3_get_app_decl(ctx, app);
    const kind = z3.Z3_get_decl_kind(ctx, decl);

    switch (kind) {
        z3.c.Z3_OP_BUDIV,
        z3.c.Z3_OP_BUDIV_I,
        => return true,
        z3.c.Z3_OP_BMUL => {
            if (!isAllOnesNegationMul(ctx, app)) return true;
        },
        else => {},
    }

    const num_args = z3.Z3_get_app_num_args(ctx, app);
    for (0..@intCast(num_args)) |arg_idx| {
        if (containsHeavyPathArithmetic(ctx, z3.Z3_get_app_arg(ctx, app, @intCast(arg_idx)))) {
            return true;
        }
    }
    return false;
}

fn isAllOnesNegationMul(ctx: z3.Z3_context, app: z3.Z3_app) bool {
    if (z3.Z3_get_app_num_args(ctx, app) != 2) return false;
    const lhs = z3.Z3_get_app_arg(ctx, app, 0);
    const rhs = z3.Z3_get_app_arg(ctx, app, 1);
    return isAllOnesBitvectorConstant(ctx, lhs) or isAllOnesBitvectorConstant(ctx, rhs);
}

fn isAllOnesBitvectorConstant(ctx: z3.Z3_context, ast: z3.Z3_ast) bool {
    if (z3.Z3_get_ast_kind(ctx, ast) != z3.Z3_APP_AST) return false;
    const sort = z3.Z3_get_sort(ctx, ast);
    if (z3.Z3_get_sort_kind(ctx, sort) != z3.Z3_BV_SORT) return false;
    const text_ptr = z3.Z3_ast_to_string(ctx, ast);
    if (text_ptr == null) return false;
    const text = std.mem.span(text_ptr);
    if (text.len < 3 or text[0] != '#' or text[1] != 'x') return false;
    for (text[2..]) |ch| {
        if (ch != 'f' and ch != 'F') return false;
    }
    return true;
}

const ActivePathAssume = struct {
    condition: z3.Z3_ast,
    extra_constraints: []const z3.Z3_ast,
    owned_extra_constraints: bool = false,
};

const EncodedAnnotation = struct {
    function_name: []const u8,
    kind: AnnotationKind,
    condition: z3.Z3_ast,
    condition_value: ?mlir.MlirValue = null,
    source_op: ?mlir.MlirOperation = null,
    extra_constraints: []const z3.Z3_ast,
    path_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    old_condition: ?z3.Z3_ast = null,
    old_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    loop_entry_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    loop_step_condition: ?z3.Z3_ast = null,
    loop_step_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    loop_step_head_condition: ?z3.Z3_ast = null,
    loop_step_head_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    loop_step_body_condition: ?z3.Z3_ast = null,
    loop_step_body_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    loop_exit_condition: ?z3.Z3_ast = null,
    loop_exit_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    file: []const u8,
    line: u32,
    column: u32,
    label: ?[]const u8 = null,
    guard_id: ?[]const u8 = null,
    loop_owner: ?u64 = null,
};

fn isAssumptionKind(kind: AnnotationKind) bool {
    return switch (kind) {
        .Requires, .Assume => true,
        else => false,
    };
}

fn isObligationKind(kind: AnnotationKind) bool {
    return switch (kind) {
        .Guard, .Ensures, .LoopInvariant, .ContractInvariant => true,
        else => false,
    };
}

fn isGlobalContractInvariantAnnotation(ann: EncodedAnnotation) bool {
    return ann.kind == .ContractInvariant and
        ann.loop_owner == null and
        std.mem.eql(u8, ann.function_name, "unknown");
}

fn obligationErrorType(kind: AnnotationKind) errors.VerificationErrorType {
    return switch (kind) {
        .Guard, .Requires => .PreconditionViolation,
        .Ensures => .PostconditionViolation,
        .LoopInvariant, .ContractInvariant => .InvariantViolation,
        else => .Unknown,
    };
}

fn obligationKindLabel(kind: AnnotationKind) []const u8 {
    return switch (kind) {
        .Guard => "guard",
        .Ensures => "ensures",
        .LoopInvariant => "invariant",
        .ContractInvariant => "contract invariant",
        else => "obligation",
    };
}

fn inferInvariantLabelFromLine(line: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, line, "invariant ")) return null;
    const rest = std.mem.trimLeft(u8, line["invariant ".len..], " \t");
    const open_paren = std.mem.indexOfScalar(u8, rest, '(') orelse return null;
    const candidate = std.mem.trim(u8, rest[0..open_paren], " \t");
    if (candidate.len == 0) return null;
    return candidate;
}

fn shortQueryHash(hash: u64) u32 {
    return @truncate(hash);
}

fn formatQueryTag(allocator: std.mem.Allocator, constraint_count: usize, smtlib_hash: u64) ![]u8 {
    return std.fmt.allocPrint(allocator, " [q={x:0>8} c={d}]", .{ shortQueryHash(smtlib_hash), constraint_count });
}

fn buildQueryMetadata(self: *VerificationPass, constraints: []const z3.Z3_ast) !struct { constraint_count: usize, smtlib_hash: u64 } {
    const built = try buildSmtlibForConstraints(self.allocator, &self.solver, constraints);
    defer self.allocator.free(built.smtlib_z);
    if (built.decl_symbols.len > 0) self.allocator.free(built.decl_symbols);
    if (built.decls.len > 0) self.allocator.free(built.decls);
    return .{
        .constraint_count = constraints.len,
        .smtlib_hash = std.hash.Wyhash.hash(0, built.smtlib_z),
    };
}

test "verification infers named invariant label from source line" {
    try std.testing.expectEqualStrings(
        "value_nonnegative",
        inferInvariantLabelFromLine("invariant value_nonnegative(value >= 0);").?,
    );
}

test "verification ignores unlabeled invariant source line" {
    try std.testing.expect(inferInvariantLabelFromLine("invariant value >= 0;") == null);
}

fn astEquivalent(self: *VerificationPass, lhs: z3.Z3_ast, rhs: z3.Z3_ast) bool {
    const eq = z3.c.Z3_is_eq_ast(self.context.ctx, lhs, rhs);
    return if (@TypeOf(eq) == bool) eq else eq != 0;
}

fn constraintSliceContains(self: *VerificationPass, haystack: []const z3.Z3_ast, needle: z3.Z3_ast) bool {
    for (haystack) |candidate| {
        if (astEquivalent(self, candidate, needle)) return true;
    }
    return false;
}

fn annotationLocationPrecedes(lhs: EncodedAnnotation, rhs: EncodedAnnotation) bool {
    if (!std.mem.eql(u8, lhs.file, rhs.file)) return false;
    if (lhs.line < rhs.line) return true;
    if (lhs.line > rhs.line) return false;
    return lhs.column <= rhs.column;
}

fn annotationKindSortKey(kind: AnnotationKind) u8 {
    return switch (kind) {
        .Requires => 0,
        .Assume => 1,
        .PathAssume => 2,
        .Guard => 3,
        .RefinementGuard => 4,
        .Ensures => 5,
        .LoopInvariant => 6,
        .ContractInvariant => 7,
    };
}

fn lessThanEncodedAnnotation(_: void, lhs: EncodedAnnotation, rhs: EncodedAnnotation) bool {
    const fn_order = std.mem.order(u8, lhs.function_name, rhs.function_name);
    if (fn_order != .eq) return fn_order == .lt;

    const file_order = std.mem.order(u8, lhs.file, rhs.file);
    if (file_order != .eq) return file_order == .lt;

    if (lhs.line != rhs.line) return lhs.line < rhs.line;
    if (lhs.column != rhs.column) return lhs.column < rhs.column;

    const lhs_kind = annotationKindSortKey(lhs.kind);
    const rhs_kind = annotationKindSortKey(rhs.kind);
    if (lhs_kind != rhs_kind) return lhs_kind < rhs_kind;

    const lhs_guard = lhs.guard_id orelse "";
    const rhs_guard = rhs.guard_id orelse "";
    const guard_order = std.mem.order(u8, lhs_guard, rhs_guard);
    if (guard_order != .eq) return guard_order == .lt;

    const lhs_loop = lhs.loop_owner orelse 0;
    const rhs_loop = rhs.loop_owner orelse 0;
    if (lhs_loop != rhs_loop) return lhs_loop < rhs_loop;

    return false;
}

fn stableSortAnnotations(items: []EncodedAnnotation) void {
    std.mem.sort(EncodedAnnotation, items, {}, lessThanEncodedAnnotation);
}

fn lessThanString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.order(u8, lhs, rhs) == .lt;
}

fn collectSortedFunctionNames(
    allocator: std.mem.Allocator,
    by_function: *const std.StringHashMap(ManagedArrayList(EncodedAnnotation)),
) !std.ArrayList([]const u8) {
    var names = std.ArrayList([]const u8){};
    errdefer names.deinit(allocator);

    var it = by_function.iterator();
    while (it.next()) |entry| {
        try names.append(allocator, entry.key_ptr.*);
    }

    std.mem.sort([]const u8, names.items, {}, lessThanString);
    return names;
}

fn pathAssumeAppliesToAnnotation(
    self: *VerificationPass,
    path_assume: EncodedAnnotation,
    ann: EncodedAnnotation,
) bool {
    if (!annotationLocationPrecedes(path_assume, ann)) return false;
    return pathConstraintsCompatible(self, path_assume.path_constraints, ann.path_constraints);
}

fn addApplicablePathAssumptionsToSolver(
    self: *VerificationPass,
    path_assumption_annotations: []const EncodedAnnotation,
    ann: EncodedAnnotation,
    relevant_symbols: ?*const ManagedArrayList([]const u8),
) !void {
    for (path_assumption_annotations) |path_assume| {
        if (!pathAssumeAppliesToAnnotation(self, path_assume, ann)) continue;
        for (path_assume.path_constraints) |cst| {
            if (relevant_symbols) |symbols| {
                if (!astUsesOnlyRelevantSymbols(self, cst, symbols)) continue;
            }
            try self.solver.assertChecked(cst);
        }
        for (path_assume.extra_constraints) |cst| {
            if (relevant_symbols) |symbols| {
                if (!astUsesOnlyRelevantSymbols(self, cst, symbols)) continue;
            }
            try self.solver.assertChecked(cst);
        }
        if (relevant_symbols) |symbols| {
            if (!astUsesOnlyRelevantSymbols(self, path_assume.condition, symbols)) continue;
        }
        try self.solver.assertChecked(path_assume.condition);
    }
}

fn addApplicablePathAssumptionsToConstraintList(
    self: *VerificationPass,
    constraints: *ManagedArrayList(z3.Z3_ast),
    path_assumption_annotations: []const EncodedAnnotation,
    ann: EncodedAnnotation,
    relevant_symbols: ?*const ManagedArrayList([]const u8),
) !void {
    for (path_assumption_annotations) |path_assume| {
        if (!pathAssumeAppliesToAnnotation(self, path_assume, ann)) continue;
        try addRelevantConstraintSlice(self, constraints, path_assume.path_constraints, relevant_symbols);
        try addRelevantConstraintSlice(self, constraints, path_assume.extra_constraints, relevant_symbols);
        if (relevant_symbols) |symbols| {
            if (!astUsesOnlyRelevantSymbols(self, path_assume.condition, symbols)) continue;
        }
        try constraints.append(path_assume.condition);
    }
}

fn pathConstraintsCompatible(self: *VerificationPass, previous_path: []const z3.Z3_ast, current_path: []const z3.Z3_ast) bool {
    // previous guard must be on a path-prefix compatible with the current guard.
    // This is approximated as: every previous path constraint appears in current path constraints.
    for (previous_path) |prev_cst| {
        if (!constraintSliceContains(self, current_path, prev_cst)) return false;
    }
    return true;
}

fn constraintSlicesEquivalent(self: *VerificationPass, lhs: []const z3.Z3_ast, rhs: []const z3.Z3_ast) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |l_ast, r_ast| {
        if (!astEquivalent(self, l_ast, r_ast)) return false;
    }
    return true;
}

fn encodedAnnotationEquivalent(self: *VerificationPass, lhs: EncodedAnnotation, rhs: EncodedAnnotation) bool {
    if (!std.mem.eql(u8, lhs.function_name, rhs.function_name)) return false;
    if (lhs.kind != rhs.kind) return false;
    if (!std.mem.eql(u8, lhs.file, rhs.file)) return false;
    if (lhs.line != rhs.line or lhs.column != rhs.column) return false;
    if ((lhs.loop_owner == null) != (rhs.loop_owner == null)) return false;
    if (lhs.loop_owner != rhs.loop_owner) return false;
    if ((lhs.guard_id == null) != (rhs.guard_id == null)) return false;
    if (lhs.guard_id) |lhs_guard| {
        if (!std.mem.eql(u8, lhs_guard, rhs.guard_id.?)) return false;
    }
    if ((lhs.label == null) != (rhs.label == null)) return false;
    if (lhs.label) |lhs_label| {
        if (!std.mem.eql(u8, lhs_label, rhs.label.?)) return false;
    }
    if ((lhs.old_condition == null) != (rhs.old_condition == null)) return false;
    if (lhs.old_condition) |lhs_old| {
        if (!astEquivalent(self, lhs_old, rhs.old_condition.?)) return false;
    }
    if ((lhs.loop_step_condition == null) != (rhs.loop_step_condition == null)) return false;
    if (lhs.loop_step_condition) |lhs_step| {
        if (!astEquivalent(self, lhs_step, rhs.loop_step_condition.?)) return false;
    }
    if ((lhs.loop_step_head_condition == null) != (rhs.loop_step_head_condition == null)) return false;
    if (lhs.loop_step_head_condition) |lhs_step_head| {
        if (!astEquivalent(self, lhs_step_head, rhs.loop_step_head_condition.?)) return false;
    }
    if ((lhs.loop_exit_condition == null) != (rhs.loop_exit_condition == null)) return false;
    if (lhs.loop_exit_condition) |lhs_exit| {
        if (!astEquivalent(self, lhs_exit, rhs.loop_exit_condition.?)) return false;
    }
    if (!astEquivalent(self, lhs.condition, rhs.condition)) return false;
    if (!constraintSlicesEquivalent(self, lhs.extra_constraints, rhs.extra_constraints)) return false;
    if (!constraintSlicesEquivalent(self, lhs.path_constraints, rhs.path_constraints)) return false;
    if (!constraintSlicesEquivalent(self, lhs.old_extra_constraints, rhs.old_extra_constraints)) return false;
    if (!constraintSlicesEquivalent(self, lhs.loop_entry_extra_constraints, rhs.loop_entry_extra_constraints)) return false;
    if (!constraintSlicesEquivalent(self, lhs.loop_step_extra_constraints, rhs.loop_step_extra_constraints)) return false;
    if (!constraintSlicesEquivalent(self, lhs.loop_step_head_extra_constraints, rhs.loop_step_head_extra_constraints)) return false;
    if (!constraintSlicesEquivalent(self, lhs.loop_exit_extra_constraints, rhs.loop_exit_extra_constraints)) return false;
    return true;
}

fn appendEncodedAnnotationUnique(self: *VerificationPass, ann: EncodedAnnotation) !void {
    for (self.encoded_annotations.items) |existing| {
        if (encodedAnnotationEquivalent(self, existing, ann)) return;
    }
    try self.encoded_annotations.append(ann);
}

fn collectAstSymbols(
    self: *VerificationPass,
    ast: z3.Z3_ast,
    symbols: *ManagedArrayList([]const u8),
) !void {
    if (z3.Z3_get_ast_kind(self.context.ctx, ast) != z3.Z3_APP_AST) return;

    const app = z3.Z3_to_app(self.context.ctx, ast);
    const num_args = z3.Z3_get_app_num_args(self.context.ctx, app);
    if (num_args == 0) {
        const decl = z3.Z3_get_app_decl(self.context.ctx, app);
        if (z3.Z3_get_decl_kind(self.context.ctx, decl) == z3.c.Z3_OP_UNINTERPRETED) {
            const symbol = z3.Z3_get_decl_name(self.context.ctx, decl);
            const symbol_text = z3.Z3_get_symbol_string(self.context.ctx, symbol);
            if (symbol_text != null) {
                const name = std.mem.span(symbol_text);
                if (!symbolListContains(symbols.items, name)) {
                    try symbols.append(name);
                }
            }
        }
        return;
    }

    for (0..@intCast(num_args)) |arg_idx| {
        try collectAstSymbols(self, z3.Z3_get_app_arg(self.context.ctx, app, @intCast(arg_idx)), symbols);
    }
}

fn astUsesOnlyRelevantSymbols(
    self: *VerificationPass,
    ast: z3.Z3_ast,
    relevant_symbols: *const ManagedArrayList([]const u8),
) bool {
    if (z3.Z3_get_ast_kind(self.context.ctx, ast) != z3.Z3_APP_AST) return true;

    const app = z3.Z3_to_app(self.context.ctx, ast);
    const num_args = z3.Z3_get_app_num_args(self.context.ctx, app);
    if (num_args == 0) {
        const decl = z3.Z3_get_app_decl(self.context.ctx, app);
        if (z3.Z3_get_decl_kind(self.context.ctx, decl) == z3.c.Z3_OP_UNINTERPRETED) {
            const symbol = z3.Z3_get_decl_name(self.context.ctx, decl);
            const symbol_text = z3.Z3_get_symbol_string(self.context.ctx, symbol);
            if (symbol_text == null) return true;
            return symbolListContains(relevant_symbols.items, std.mem.span(symbol_text));
        }
        return true;
    }

    for (0..@intCast(num_args)) |arg_idx| {
        if (!astUsesOnlyRelevantSymbols(self, z3.Z3_get_app_arg(self.context.ctx, app, @intCast(arg_idx)), relevant_symbols)) {
            return false;
        }
    }
    return true;
}

fn extractZeroArgUninterpretedSymbol(self: *VerificationPass, ast: z3.Z3_ast) ?[]const u8 {
    if (z3.Z3_get_ast_kind(self.context.ctx, ast) != z3.Z3_APP_AST) return null;
    const app = z3.Z3_to_app(self.context.ctx, ast);
    if (z3.Z3_get_app_num_args(self.context.ctx, app) != 0) return null;
    const decl = z3.Z3_get_app_decl(self.context.ctx, app);
    if (z3.Z3_get_decl_kind(self.context.ctx, decl) != z3.c.Z3_OP_UNINTERPRETED) return null;
    const symbol = z3.Z3_get_decl_name(self.context.ctx, decl);
    const symbol_text = z3.Z3_get_symbol_string(self.context.ctx, symbol);
    if (symbol_text == null) return null;
    return std.mem.span(symbol_text);
}

fn expandRelevantSymbolsFromSimpleEqualities(
    self: *VerificationPass,
    symbols: *ManagedArrayList([]const u8),
    constraints: []const z3.Z3_ast,
) !void {
    var changed = true;
    while (changed) {
        changed = false;
        for (constraints) |constraint| {
            if (z3.Z3_get_ast_kind(self.context.ctx, constraint) != z3.Z3_APP_AST) continue;
            const app = z3.Z3_to_app(self.context.ctx, constraint);
            if (z3.Z3_get_app_num_args(self.context.ctx, app) != 2) continue;
            const decl = z3.Z3_get_app_decl(self.context.ctx, app);
            if (z3.Z3_get_decl_kind(self.context.ctx, decl) != z3.c.Z3_OP_EQ) continue;

            const lhs_symbol = extractZeroArgUninterpretedSymbol(self, z3.Z3_get_app_arg(self.context.ctx, app, 0));
            const rhs_symbol = extractZeroArgUninterpretedSymbol(self, z3.Z3_get_app_arg(self.context.ctx, app, 1));

            if (lhs_symbol) |lhs| {
                if (rhs_symbol) |rhs| {
                    const lhs_known = symbolListContains(symbols.items, lhs);
                    const rhs_known = symbolListContains(symbols.items, rhs);
                    if (lhs_known and !rhs_known) {
                        try symbols.append(rhs);
                        changed = true;
                    } else if (rhs_known and !lhs_known) {
                        try symbols.append(lhs);
                        changed = true;
                    }
                }
            }
        }
    }
}

fn buildRelevantSymbolSetForAnnotation(
    self: *VerificationPass,
    ann: EncodedAnnotation,
) !ManagedArrayList([]const u8) {
    var symbols = ManagedArrayList([]const u8).init(self.allocator);
    errdefer symbols.deinit();

    try collectAstSymbols(self, ann.condition, &symbols);
    for (ann.path_constraints) |path_constraint| {
        try collectAstSymbols(self, path_constraint, &symbols);
    }
    for (ann.extra_constraints) |extra_constraint| {
        try collectAstSymbols(self, extra_constraint, &symbols);
    }
    if (ann.old_condition) |old_cond| {
        try collectAstSymbols(self, old_cond, &symbols);
    }
    if (ann.loop_step_condition) |step_cond| {
        try collectAstSymbols(self, step_cond, &symbols);
    }
    if (ann.loop_exit_condition) |exit_cond| {
        try collectAstSymbols(self, exit_cond, &symbols);
    }
    try expandRelevantSymbolsFromSimpleEqualities(self, &symbols, ann.path_constraints);
    try expandRelevantSymbolsFromSimpleEqualities(self, &symbols, ann.extra_constraints);

    return symbols;
}

fn addRelevantConstraintSlice(
    self: *VerificationPass,
    list: *ManagedArrayList(z3.Z3_ast),
    constraints: []const z3.Z3_ast,
    relevant_symbols: ?*const ManagedArrayList([]const u8),
) !void {
    for (constraints) |constraint| {
        if (relevant_symbols) |symbols| {
            if (!astUsesOnlyRelevantSymbols(self, constraint, symbols)) continue;
        }
        try list.append(constraint);
    }
}

fn addRelevantOldLinkConstraints(
    self: *VerificationPass,
    list: *ManagedArrayList(z3.Z3_ast),
    relevant_symbols: *const ManagedArrayList([]const u8),
) !void {
    var old_it = self.encoder.global_old_map.iterator();
    while (old_it.next()) |entry| {
        const base_name = entry.key_ptr.*;
        const old_ast = entry.value_ptr.*;
        const old_symbol = extractZeroArgUninterpretedSymbol(self, old_ast) orelse continue;
        if (!symbolListContains(relevant_symbols.items, old_symbol)) continue;

        const entry_ast = self.encoder.global_entry_map.get(base_name) orelse continue;
        const eq = z3.Z3_mk_eq(self.context.ctx, old_ast, entry_ast);
        if (constraintSliceContains(self, list.items, eq)) continue;
        try list.append(eq);
    }
}

fn addExplicitOldLinkConstraintsForAst(
    self: *VerificationPass,
    list: *ManagedArrayList(z3.Z3_ast),
    ast: z3.Z3_ast,
) !void {
    var symbols = ManagedArrayList([]const u8).init(self.allocator);
    defer symbols.deinit();
    try collectAstSymbols(self, ast, &symbols);
    try addRelevantOldLinkConstraints(self, list, &symbols);
}

fn symbolListContains(symbols: []const []const u8, needle: []const u8) bool {
    for (symbols) |symbol| {
        if (std.mem.eql(u8, symbol, needle)) return true;
    }
    return false;
}

fn sameLoopInvariantGroup(self: *VerificationPass, reference: EncodedAnnotation, candidate: EncodedAnnotation) bool {
    if (reference.loop_owner != null and candidate.loop_owner != null) {
        return reference.loop_owner == candidate.loop_owner;
    }
    const ref_exit = reference.loop_exit_condition orelse return false;
    const cand_exit = candidate.loop_exit_condition orelse return false;
    if (!astEquivalent(self, ref_exit, cand_exit)) return false;
    return constraintSlicesEquivalent(self, reference.path_constraints, candidate.path_constraints);
}

const QueryKind = enum {
    Base,
    Obligation,
    LoopInvariantStep,
    LoopInvariantPost,
    GuardSatisfy,
    GuardViolate,
};

const ReportQueryRun = struct {
    status: z3.Z3_lbool = z3.Z3_L_UNDEF,
    elapsed_ms: u64 = 0,
    model: ?[]u8 = null,
    explain: ?[]u8 = null,
    explain_tags: []const AssumptionTag = &.{},
    vacuous: bool = false,
};

const ReportSummary = struct {
    total_queries: u64 = 0,
    sat: u64 = 0,
    unsat: u64 = 0,
    unknown: u64 = 0,
    vacuous: u64 = 0,
    failed_obligations: u64 = 0,
    inconsistent_bases: u64 = 0,
    proven_guards: u64 = 0,
    violatable_guards: u64 = 0,
    verification_success: bool = true,
    verification_errors: u64 = 0,
    verification_diagnostics: u64 = 0,
    encoding_degraded: bool = false,
    degradation_reason: ?[]const u8 = null,
};

const ReportKindCounts = struct {
    base: u64 = 0,
    obligation: u64 = 0,
    loop_invariant_step: u64 = 0,
    loop_invariant_post: u64 = 0,
    guard_satisfy: u64 = 0,
    guard_violate: u64 = 0,
};

fn verifyModeLabel(mode: VerificationPass.VerifyMode) []const u8 {
    return switch (mode) {
        .Basic => "basic",
        .Full => "full",
    };
}

fn queryKindLabel(kind: QueryKind) []const u8 {
    return switch (kind) {
        .Base => "base",
        .Obligation => "obligation",
        .LoopInvariantStep => "loop_invariant_step",
        .LoopInvariantPost => "loop_invariant_post",
        .GuardSatisfy => "guard_satisfy",
        .GuardViolate => "guard_violate",
    };
}

fn queryStatusLabel(status: z3.Z3_lbool) []const u8 {
    return switch (status) {
        z3.Z3_L_TRUE => "SAT",
        z3.Z3_L_FALSE => "UNSAT",
        else => "UNKNOWN",
    };
}

const FailureClassification = struct {
    subtype: ?[]const u8 = null,
    confidence: ?[]const u8 = null,
    evidence: ?[]const u8 = null,
    narrowing_bits: ?u32 = null,
    refinement_kind: ?[]const u8 = null,
};

fn shouldHideCounterexampleVariable(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "undef_") or
        std.mem.startsWith(u8, name, "old_") or
        std.mem.startsWith(u8, name, "__ora_");
}

fn filePathsLikelyMatch(a: []const u8, b: []const u8) bool {
    if (a.len == 0 or b.len == 0) return true;
    if (std.mem.eql(u8, a, b)) return true;
    return std.mem.indexOf(u8, a, b) != null or std.mem.indexOf(u8, b, a) != null;
}

fn parseGuardRefinementKind(guard_id: []const u8) ?[]const u8 {
    const last_colon = std.mem.lastIndexOfScalar(u8, guard_id, ':') orelse return null;
    if (last_colon == 0) return null;
    const before_last = guard_id[0..last_colon];
    const prev_colon = std.mem.lastIndexOfScalar(u8, before_last, ':') orelse return null;
    if (prev_colon + 1 >= last_colon) return null;
    return before_last[prev_colon + 1 ..];
}

fn classifyGuardId(guard_id: []const u8, violatable: bool) FailureClassification {
    const kind = parseGuardRefinementKind(guard_id);
    var classification = FailureClassification{
        .confidence = "high",
        .evidence = guard_id,
        .refinement_kind = kind,
    };

    if (violatable) {
        classification.subtype = if (kind) |k|
            if (std.mem.eql(u8, k, "non_zero_address"))
                "GuardViolation.NonZeroAddress"
            else if (std.mem.eql(u8, k, "min_value"))
                "GuardViolation.MinValue"
            else
                "GuardViolation"
        else
            "GuardViolation";
    } else {
        classification.subtype = if (kind) |k|
            if (std.mem.eql(u8, k, "non_zero_address"))
                "GuardUnsatisfiable.NonZeroAddress"
            else if (std.mem.eql(u8, k, "min_value"))
                "GuardUnsatisfiable.MinValue"
            else
                "GuardUnsatisfiable"
        else
            "GuardUnsatisfiable";
    }

    return classification;
}

fn inferNarrowingBitsFromSmtlib(smtlib: []const u8) ?u32 {
    const marker = "(_ extract ";
    const start = std.mem.indexOf(u8, smtlib, marker) orelse return null;
    const digits_start = start + marker.len;
    if (digits_start >= smtlib.len) return null;

    var end = digits_start;
    while (end < smtlib.len and std.ascii.isDigit(smtlib[end])) : (end += 1) {}
    if (end == digits_start) return null;

    const high = std.fmt.parseInt(u32, smtlib[digits_start..end], 10) catch return null;
    return high + 1;
}

fn classifyArithmeticPatternFromSmtlib(smtlib: []const u8) FailureClassification {
    if (std.mem.indexOf(u8, smtlib, "(_ zero_extend ") != null and
        std.mem.indexOf(u8, smtlib, "(_ extract ") != null)
    {
        return .{
            .subtype = "NarrowingOverflow",
            .confidence = "high",
            .evidence = "pattern: zero_extend + extract narrowing check",
            .narrowing_bits = inferNarrowingBitsFromSmtlib(smtlib),
        };
    }

    if (std.mem.indexOf(u8, smtlib, "(bvadd") != null and
        std.mem.indexOf(u8, smtlib, "(bvult") != null)
    {
        return .{
            .subtype = "AdditionOverflow",
            .confidence = "high",
            .evidence = "pattern: bvadd + bvult overflow guard",
        };
    }

    if (std.mem.indexOf(u8, smtlib, "(not (xor (bvult") != null and
        std.mem.indexOf(u8, smtlib, "(bvadd") == null and
        std.mem.indexOf(u8, smtlib, "(_ zero_extend ") == null)
    {
        return .{
            .subtype = "SubtractionUnderflow",
            .confidence = "medium",
            .evidence = "pattern: negated unsigned less-than subtraction guard",
        };
    }

    if (std.mem.indexOf(u8, smtlib, "(bvmul") != null and
        (std.mem.indexOf(u8, smtlib, "(bvudiv") != null or std.mem.indexOf(u8, smtlib, "(bvsdiv") != null))
    {
        return .{
            .subtype = "MultiplicationOverflow",
            .confidence = "medium",
            .evidence = "pattern: multiply/divide overflow witness",
        };
    }

    return .{};
}

fn classifyQueryFailure(query: PreparedQuery, run: ReportQueryRun) FailureClassification {
    return switch (query.kind) {
        .Base => if (run.status == z3.Z3_L_FALSE)
            .{
                .subtype = "InconsistentAssumptions",
                .confidence = "high",
                .evidence = "base query is UNSAT",
            }
        else if (run.status == z3.Z3_L_UNDEF)
            .{
                .subtype = "UnknownAssumptions",
                .confidence = "high",
                .evidence = "base query returned UNKNOWN",
            }
        else
            .{},
        .GuardSatisfy => if (run.status == z3.Z3_L_FALSE and query.guard_id != null)
            classifyGuardId(query.guard_id.?, false)
        else if (run.status == z3.Z3_L_UNDEF)
            .{
                .subtype = "UnknownGuardSatisfiability",
                .confidence = "high",
                .evidence = "guard satisfiability query returned UNKNOWN",
            }
        else
            .{},
        .GuardViolate => if (run.status == z3.Z3_L_TRUE and query.guard_id != null)
            classifyGuardId(query.guard_id.?, true)
        else if (run.status == z3.Z3_L_UNDEF)
            .{
                .subtype = "UnknownGuardRemovability",
                .confidence = "high",
                .evidence = "guard removability query returned UNKNOWN",
            }
        else
            .{},
        .Obligation, .LoopInvariantStep, .LoopInvariantPost => blk: {
            if (run.status == z3.Z3_L_UNDEF) break :blk .{
                .subtype = switch (query.kind) {
                    .LoopInvariantStep => "UnknownLoopInvariantStep",
                    .LoopInvariantPost => "UnknownLoopPostcondition",
                    else => "UnknownObligation",
                },
                .confidence = "high",
                .evidence = "obligation query returned UNKNOWN",
            };
            if (run.status != z3.Z3_L_TRUE) break :blk .{};
            const arithmetic = classifyArithmeticPatternFromSmtlib(query.smtlib_z);
            if (arithmetic.subtype != null) break :blk arithmetic;
            break :blk .{
                .subtype = switch (query.obligation_kind orelse .ContractInvariant) {
                    .Guard => "GuardViolation",
                    .Requires => "PreconditionViolation",
                    .Ensures => "PostconditionViolation",
                    .LoopInvariant => "LoopInvariantViolation",
                    .ContractInvariant => "ContractInvariantViolation",
                    else => "ObligationViolation",
                },
                .confidence = "low",
                .evidence = "no arithmetic pattern matched",
            };
        },
    };
}

fn queryMatchesError(err: errors.VerificationError, query: PreparedQuery, run: ReportQueryRun) bool {
    if (err.line != query.line or err.column != query.column) return false;
    if (!filePathsLikelyMatch(err.file, query.file)) return false;

    return switch (err.error_type) {
        .PreconditionViolation => query.kind == .Base and run.status == z3.Z3_L_FALSE,
        .RefinementViolation => query.kind == .GuardSatisfy and run.status == z3.Z3_L_FALSE,
        .InvariantViolation, .PostconditionViolation, .ArithmeticOverflow, .ArithmeticUnderflow, .DivisionByZero => (query.kind == .Obligation or query.kind == .LoopInvariantStep or query.kind == .LoopInvariantPost) and run.status == z3.Z3_L_TRUE,
        .Unknown => switch (query.kind) {
            .Base => run.status == z3.Z3_L_UNDEF,
            .Obligation, .LoopInvariantStep, .LoopInvariantPost, .GuardSatisfy, .GuardViolate => run.status == z3.Z3_L_UNDEF,
        },
        else => false,
    };
}

fn findQueryIndexForError(
    err: errors.VerificationError,
    queries: []const PreparedQuery,
    runs: []const ReportQueryRun,
) ?usize {
    var best: ?usize = null;
    for (queries, runs, 0..) |query, run, idx| {
        if (!queryMatchesError(err, query, run)) continue;
        if (best == null) {
            best = idx;
            continue;
        }
        // Prefer exact file match when multiple queries share the same line.
        const current_best = best.?;
        if (std.mem.eql(u8, err.file, query.file) and !std.mem.eql(u8, err.file, queries[current_best].file)) {
            best = idx;
        }
    }
    return best;
}

fn findQueryIndexForDiagnostic(
    diag: errors.Diagnostic,
    queries: []const PreparedQuery,
    runs: []const ReportQueryRun,
) ?usize {
    for (queries, runs, 0..) |query, run, idx| {
        if (query.kind != .GuardViolate or run.status != z3.Z3_L_TRUE) continue;
        if (query.guard_id == null) continue;
        if (!std.mem.eql(u8, query.guard_id.?, diag.guard_id)) continue;
        if (!std.mem.eql(u8, query.function_name, diag.function_name)) continue;
        return idx;
    }
    return null;
}

fn writeJsonStringEscaped(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try writer.print("\\u{X:0>4}", .{@as(u32, ch)});
                } else {
                    try writer.writeByte(ch);
                }
            },
        }
    }
    try writer.writeByte('"');
}

fn writeAssumptionTagJson(writer: anytype, tag: AssumptionTag) !void {
    try writer.writeByte('{');
    try writer.writeAll("\"kind\":");
    try writeJsonStringEscaped(writer, @tagName(tag.kind));
    try writer.writeAll(",\"function_name\":");
    try writeJsonStringEscaped(writer, tag.function_name);
    try writer.writeAll(",\"file\":");
    try writeJsonStringEscaped(writer, tag.file);
    try writer.print(",\"line\":{d}", .{tag.line});
    try writer.print(",\"column\":{d}", .{tag.column});
    try writer.writeAll(",\"label\":");
    try writeJsonStringEscaped(writer, tag.label);
    try writer.writeAll(",\"guard_id\":");
    if (tag.guard_id) |guard_id| {
        try writeJsonStringEscaped(writer, guard_id);
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll(",\"loop_owner\":");
    if (tag.loop_owner) |loop_owner| {
        try writer.print("{d}", .{loop_owner});
    } else {
        try writer.writeAll("null");
    }
    try writer.writeByte('}');
}

fn collectSortedStringKeys(
    allocator: std.mem.Allocator,
    map: anytype,
) !std.ArrayList([]const u8) {
    var keys = std.ArrayList([]const u8){};
    errdefer keys.deinit(allocator);

    var it = map.iterator();
    while (it.next()) |entry| {
        try keys.append(allocator, entry.key_ptr.*);
    }

    std.mem.sort([]const u8, keys.items, {}, lessThanString);
    return keys;
}

fn writeCounterexampleMarkdown(writer: anytype, allocator: std.mem.Allocator, ce: errors.Counterexample) !void {
    var keys = try collectSortedStringKeys(allocator, ce.variables);
    defer keys.deinit(allocator);

    for (keys.items) |key| {
        const value = ce.variables.get(key).?;
        try writer.print("  - `{s} = {s}`\n", .{ key, value });
    }
}

fn writeCounterexampleJson(writer: anytype, allocator: std.mem.Allocator, ce: errors.Counterexample) !void {
    var keys = try collectSortedStringKeys(allocator, ce.variables);
    defer keys.deinit(allocator);

    try writer.writeByte('{');
    var first = true;
    for (keys.items) |key| {
        if (!first) try writer.writeByte(',');
        first = false;
        try writeJsonStringEscaped(writer, key);
        try writer.writeByte(':');
        try writeJsonStringEscaped(writer, ce.variables.get(key).?);
    }
    try writer.writeByte('}');
}

const PreparedQuery = struct {
    kind: QueryKind,
    function_name: []const u8,
    guard_id: ?[]const u8 = null,
    obligation_kind: ?AnnotationKind = null,
    file: []const u8,
    line: u32,
    column: u32,
    constraints: []const z3.Z3_ast = &.{},
    tracked_assumptions: []const TrackedAssumption = &.{},
    smtlib_z: [:0]const u8,
    decl_symbols: []const z3.Z3_symbol = &.{},
    decls: []const z3.Z3_func_decl = &.{},
    constraint_count: usize = 0,
    smtlib_bytes: usize = 0,
    smtlib_hash: u64 = 0,
    log_prefix: []const u8,

    fn deinit(self: *PreparedQuery, allocator: std.mem.Allocator) void {
        if (self.constraints.len > 0) allocator.free(self.constraints);
        if (self.tracked_assumptions.len > 0) allocator.free(self.tracked_assumptions);
        allocator.free(self.smtlib_z);
        if (self.decl_symbols.len > 0) allocator.free(self.decl_symbols);
        if (self.decls.len > 0) allocator.free(self.decls);
        allocator.free(self.log_prefix);
    }
};

fn preparedQueryEquivalent(lhs: PreparedQuery, rhs: PreparedQuery) bool {
    if (lhs.kind != rhs.kind) return false;
    if (!std.mem.eql(u8, lhs.function_name, rhs.function_name)) return false;
    if (lhs.obligation_kind != rhs.obligation_kind) return false;
    if ((lhs.guard_id == null) != (rhs.guard_id == null)) return false;
    if (lhs.guard_id) |lhs_guard| {
        if (!std.mem.eql(u8, lhs_guard, rhs.guard_id.?)) return false;
    }
    if (!std.mem.eql(u8, lhs.file, rhs.file)) return false;
    if (lhs.line != rhs.line or lhs.column != rhs.column) return false;
    if (lhs.smtlib_hash != rhs.smtlib_hash) return false;
    if (lhs.tracked_assumptions.len != rhs.tracked_assumptions.len) return false;
    return std.mem.eql(u8, lhs.smtlib_z, rhs.smtlib_z);
}

fn appendPreparedQueryUnique(
    queries: *ManagedArrayList(PreparedQuery),
    allocator: std.mem.Allocator,
    query: PreparedQuery,
) !void {
    for (queries.items) |existing| {
        if (preparedQueryEquivalent(existing, query)) {
            var owned = query;
            owned.deinit(allocator);
            return;
        }
    }
    try queries.append(query);
}

const PreparedQueryResult = struct {
    status: z3.Z3_lbool = z3.Z3_L_UNDEF,
    elapsed_ms: u64 = 0,
    err: ?anyerror = null,
    model_str: ?[]const u8 = null,
    explain_str: ?[]const u8 = null,
    explain_tags: []const AssumptionTag = &.{},
    vacuous: bool = false,
};

fn scaledParallelTimeoutMs(timeout_ms: ?u32, worker_count: usize) ?u32 {
    const base = timeout_ms orelse return null;
    if (worker_count <= 1) return base;

    const widened: u64 = @as(u64, base) * @as(u64, @intCast(worker_count));
    return if (widened > std.math.maxInt(u32))
        std.math.maxInt(u32)
    else
        @as(u32, @intCast(widened));
}

fn inferReportVerificationSuccess(
    summary: ReportSummary,
    verification_result: ?*const errors.VerificationResult,
    encoding_degraded: bool,
) bool {
    if (encoding_degraded) return false;
    if (verification_result) |vr| return vr.success;
    return summary.failed_obligations == 0 and
        summary.inconsistent_bases == 0 and
        summary.unknown == 0;
}

fn parseBoolEnv(value: []const u8) bool {
    if (std.mem.eql(u8, value, "1")) return true;
    if (std.ascii.eqlIgnoreCase(value, "true")) return true;
    if (std.ascii.eqlIgnoreCase(value, "yes")) return true;
    return false;
}

fn parseVerifyMode(value: []const u8) VerificationPass.VerifyMode {
    if (std.ascii.eqlIgnoreCase(value, "full")) return .Full;
    return .Basic;
}

fn collectFunctionNames(allocator: std.mem.Allocator, mlir_module: mlir.MlirModule) !ManagedArrayList([]const u8) {
    var names = ManagedArrayList([]const u8).init(allocator);
    var seen = std.StringHashMap(void).init(allocator);
    defer {
        var it = seen.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        seen.deinit();
    }

    const module_op = mlir.oraModuleGetOperation(mlir_module);
    const num_regions = mlir.oraOperationGetNumRegions(module_op);
    for (0..@intCast(num_regions)) |region_idx| {
        const region = mlir.oraOperationGetRegion(module_op, @intCast(region_idx));
        try collectFunctionNamesInRegion(allocator, region, &names, &seen);
    }

    std.mem.sort([]const u8, names.items, {}, lessThanString);

    return names;
}

fn collectFunctionNamesInRegion(
    allocator: std.mem.Allocator,
    region: mlir.MlirRegion,
    names: *ManagedArrayList([]const u8),
    seen: *std.StringHashMap(void),
) !void {
    var current_block = mlir.oraRegionGetFirstBlock(region);
    while (!mlir.oraBlockIsNull(current_block)) {
        var current_op = mlir.oraBlockGetFirstOperation(current_block);
        while (!mlir.oraOperationIsNull(current_op)) {
            const op_name_ref = mlir.oraOperationGetName(current_op);
            defer @import("mlir_c_api").freeStringRef(op_name_ref);
            const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
                ""
            else
                op_name_ref.data[0..op_name_ref.length];

            if (std.mem.eql(u8, op_name, "func.func")) {
                const name_attr = mlir.oraOperationGetAttributeByName(current_op, mlir.oraStringRefCreate("sym_name", 8));
                if (!mlir.oraAttributeIsNull(name_attr)) {
                    const name_ref = mlir.oraStringAttrGetValue(name_attr);
                    if (name_ref.data != null and name_ref.length > 0) {
                        const name_slice = name_ref.data[0..name_ref.length];
                        if (!seen.contains(name_slice)) {
                            const name_copy = try allocator.dupe(u8, name_slice);
                            try seen.put(name_copy, {});
                            try names.append(name_copy);
                        }
                    }
                }
            }

            const num_regions = mlir.oraOperationGetNumRegions(current_op);
            for (0..@intCast(num_regions)) |region_idx| {
                const nested_region = mlir.oraOperationGetRegion(current_op, @intCast(region_idx));
                try collectFunctionNamesInRegion(allocator, nested_region, names, seen);
            }

            current_op = mlir.oraOperationGetNextInBlock(current_op);
        }
        current_block = mlir.oraBlockGetNextInRegion(current_block);
    }
}

fn mergeVerificationResults(
    allocator: std.mem.Allocator,
    dest: *errors.VerificationResult,
    src: *errors.VerificationResult,
) !void {
    if (!src.success) {
        dest.success = false;
    }

    for (src.errors.items) |err| {
        const cloned = try cloneVerificationError(allocator, err);
        try dest.addError(cloned);
    }

    var it = src.proven_guard_ids.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!dest.proven_guard_ids.contains(key)) {
            const key_copy = try allocator.dupe(u8, key);
            try dest.proven_guard_ids.put(key_copy, {});
        }
    }
}

fn cloneVerificationError(allocator: std.mem.Allocator, err: errors.VerificationError) !errors.VerificationError {
    const message = try allocator.dupe(u8, err.message);
    errdefer allocator.free(message);
    const file = try allocator.dupe(u8, err.file);
    errdefer allocator.free(file);
    const counterexample = if (err.counterexample) |ce| try cloneCounterexample(allocator, ce) else null;

    return errors.VerificationError{
        .error_type = err.error_type,
        .message = message,
        .file = file,
        .line = err.line,
        .column = err.column,
        .counterexample = counterexample,
        .allocator = allocator,
    };
}

fn cloneCounterexample(allocator: std.mem.Allocator, ce: errors.Counterexample) !errors.Counterexample {
    var copy = errors.Counterexample.init(allocator);
    var it = ce.variables.iterator();
    while (it.next()) |entry| {
        try copy.addVariable(entry.key_ptr.*, entry.value_ptr.*);
    }
    return copy;
}

fn addConstraintSlice(list: *ManagedArrayList(z3.Z3_ast), constraints: []const z3.Z3_ast) !void {
    for (constraints) |cst| {
        try list.append(cst);
    }
}

fn cloneConstraintAstSlice(allocator: std.mem.Allocator, constraints: []const z3.Z3_ast) ![]const z3.Z3_ast {
    return if (constraints.len == 0) &.{} else try allocator.dupe(z3.Z3_ast, constraints);
}

fn cloneTrackedAssumptionSlice(
    allocator: std.mem.Allocator,
    assumptions: []const TrackedAssumption,
) ![]const TrackedAssumption {
    return if (assumptions.len == 0) &.{} else try allocator.dupe(TrackedAssumption, assumptions);
}

fn defaultTrackedAssumptionLabel(kind: AssumptionKind) []const u8 {
    return switch (kind) {
        .requires => "requires",
        .loop_invariant => "loop invariant",
        .path_assume => "path assumption",
        .goal => "goal",
    };
}

fn trackedAssumptionLabel(kind: AssumptionKind, ann: EncodedAnnotation) []const u8 {
    if (ann.label) |label| return label;
    return defaultTrackedAssumptionLabel(kind);
}

fn makeTrackedAssumptionTag(kind: AssumptionKind, ann: EncodedAnnotation) AssumptionTag {
    return .{
        .kind = kind,
        .function_name = ann.function_name,
        .file = ann.file,
        .line = ann.line,
        .column = ann.column,
        .label = trackedAssumptionLabel(kind, ann),
        .guard_id = ann.guard_id,
        .loop_owner = ann.loop_owner,
    };
}

fn appendTrackedAssumption(
    list: *ManagedArrayList(TrackedAssumption),
    ast: z3.Z3_ast,
    tag: AssumptionTag,
) !void {
    try list.append(.{
        .ast = ast,
        .tag = tag,
    });
}

fn addTrackedBaseAssumptions(
    list: *ManagedArrayList(TrackedAssumption),
    annotations: []const EncodedAnnotation,
) !void {
    for (annotations) |ann| {
        switch (ann.kind) {
            .Requires => try appendTrackedAssumption(list, ann.condition, makeTrackedAssumptionTag(.requires, ann)),
            else => {},
        }
    }
}

fn addApplicableTrackedPathAssumptions(
    self: *VerificationPass,
    list: *ManagedArrayList(TrackedAssumption),
    path_assumption_annotations: []const EncodedAnnotation,
    ann: EncodedAnnotation,
    relevant_symbols: ?*const ManagedArrayList([]const u8),
) !void {
    for (path_assumption_annotations) |path_assume| {
        if (!pathAssumeAppliesToAnnotation(self, path_assume, ann)) continue;
        if (relevant_symbols) |symbols| {
            if (!astUsesOnlyRelevantSymbols(self, path_assume.condition, symbols)) continue;
        }
        try appendTrackedAssumption(list, path_assume.condition, makeTrackedAssumptionTag(.path_assume, path_assume));
    }
}

fn appendGoalTrackedAssumption(
    list: *ManagedArrayList(TrackedAssumption),
    function_name: []const u8,
    ann: EncodedAnnotation,
    goal_ast: z3.Z3_ast,
) !void {
    try appendTrackedAssumption(list, goal_ast, .{
        .kind = .goal,
        .function_name = function_name,
        .file = ann.file,
        .line = ann.line,
        .column = ann.column,
        .label = trackedAssumptionLabel(.goal, ann),
        .guard_id = ann.guard_id,
        .loop_owner = ann.loop_owner,
    });
}

fn assertPreparedQueryConstraints(solver: *Solver, constraints: []const z3.Z3_ast) !void {
    for (constraints) |constraint| {
        try solver.assertChecked(constraint);
    }
}

fn trackedAssumptionContainsAst(tracked_assumptions: []const TrackedAssumption, ast: z3.Z3_ast) bool {
    for (tracked_assumptions) |tracked| {
        if (tracked.ast == ast) return true;
    }
    return false;
}

fn assertPreparedQueryUntrackedConstraints(solver: *Solver, query: PreparedQuery) !void {
    for (query.constraints) |constraint| {
        if (trackedAssumptionContainsAst(query.tracked_assumptions, constraint)) continue;
        try solver.assertChecked(constraint);
    }
}

fn formatAssumptionTag(
    allocator: std.mem.Allocator,
    tag: AssumptionTag,
) ![]const u8 {
    const file_name = if (tag.file.len > 0) std.fs.path.basename(tag.file) else "";
    if (file_name.len > 0 and tag.line > 0) {
        return try std.fmt.allocPrint(
            allocator,
            "{s} {s}:{d} {s}",
            .{ defaultTrackedAssumptionLabel(tag.kind), file_name, tag.line, tag.label },
        );
    }
    if (tag.line > 0) {
        return try std.fmt.allocPrint(
            allocator,
            "{s} line {d} {s}",
            .{ defaultTrackedAssumptionLabel(tag.kind), tag.line, tag.label },
        );
    }
    return try std.fmt.allocPrint(
        allocator,
        "{s} {s}",
        .{ defaultTrackedAssumptionLabel(tag.kind), tag.label },
    );
}

fn formatUnsatCoreSummary(
    allocator: std.mem.Allocator,
    tracked_assumptions: []const TrackedAssumption,
    core_asts: []const z3.Z3_ast,
) !struct { explain_str: ?[]u8, explain_tags: []const AssumptionTag } {
    if (core_asts.len == 0) return .{ .explain_str = null, .explain_tags = &.{} };

    var parts = ManagedArrayList([]const u8).init(allocator);
    defer {
        for (parts.items) |part| allocator.free(part);
        parts.deinit();
    }
    var tags = ManagedArrayList(AssumptionTag).init(allocator);
    defer tags.deinit();

    for (tracked_assumptions) |tracked| {
        const proxy = tracked.proxy orelse continue;
        var in_core = false;
        for (core_asts) |core_ast| {
            if (core_ast == proxy) {
                in_core = true;
                break;
            }
        }
        if (!in_core) continue;
        try parts.append(try formatAssumptionTag(allocator, tracked.tag));
        try tags.append(tracked.tag);
    }

    if (parts.items.len == 0) {
        return .{ .explain_str = null, .explain_tags = &.{} };
    }

    var buffer = ManagedArrayList(u8).init(allocator);
    errdefer buffer.deinit();
    const writer = buffer.writer();
    for (parts.items, 0..) |part, idx| {
        if (idx != 0) try writer.writeAll("; ");
        try writer.writeAll(part);
    }
    return .{
        .explain_str = try buffer.toOwnedSlice(),
        .explain_tags = try cloneAssumptionTagSlice(allocator, tags.items),
    };
}

const ExplainCheckResult = struct {
    status: z3.Z3_lbool,
    explain_str: ?[]u8 = null,
    explain_tags: []const AssumptionTag = &.{},
};

fn checkPreparedQueryTrackedAssumptions(
    self: *VerificationPass,
    query: PreparedQuery,
    include_goal: bool,
) !ExplainCheckResult {
    var tracked = ManagedArrayList(TrackedAssumption).init(self.allocator);
    defer tracked.deinit();
    var proxies = ManagedArrayList(z3.Z3_ast).init(self.allocator);
    defer proxies.deinit();

    for (query.tracked_assumptions) |tracked_assumption| {
        if (!include_goal and tracked_assumption.tag.kind == .goal) continue;

        const proxy = try self.solver.mkFreshBoolProxy("ora_assumption");
        const implication = z3.Z3_mk_implies(
            self.context.ctx,
            proxy,
            self.encoder.coerceBoolean(tracked_assumption.ast),
        );
        try self.solver.assertChecked(implication);

        var materialized = tracked_assumption;
        materialized.proxy = proxy;
        try tracked.append(materialized);
        try proxies.append(proxy);
    }

    if (proxies.items.len == 0) {
        return .{
            .status = try self.solver.checkChecked(),
        };
    }

    const status = try self.solver.checkAssumptionsChecked(proxies.items);
    if (status != z3.Z3_L_FALSE) {
        return .{ .status = status };
    }

    const core_asts = try self.solver.getUnsatCoreOwned();
    defer self.allocator.free(core_asts);
    const formatted = try formatUnsatCoreSummary(self.allocator, tracked.items, core_asts);
    return .{
        .status = status,
        .explain_str = formatted.explain_str,
        .explain_tags = formatted.explain_tags,
    };
}

const SmtSymbolDecl = struct {
    symbol: z3.Z3_symbol,
    decl: z3.Z3_func_decl,
};

fn symbolDeclListContains(symbols: []const SmtSymbolDecl, decl: z3.Z3_func_decl) bool {
    for (symbols) |symbol| {
        if (symbol.decl == decl) return true;
    }
    return false;
}

fn collectAstSymbolDecls(
    ctx: z3.Z3_context,
    ast: z3.Z3_ast,
    symbols: *ManagedArrayList(SmtSymbolDecl),
) !void {
    if (z3.Z3_get_ast_kind(ctx, ast) != z3.Z3_APP_AST) return;

    const app = z3.Z3_to_app(ctx, ast);
    const num_args = z3.Z3_get_app_num_args(ctx, app);
    if (num_args == 0) {
        const decl = z3.Z3_get_app_decl(ctx, app);
        if (z3.Z3_get_decl_kind(ctx, decl) == z3.c.Z3_OP_UNINTERPRETED) {
            const symbol = z3.Z3_get_decl_name(ctx, decl);
            if (!symbolDeclListContains(symbols.items, decl)) {
                try symbols.append(.{
                    .symbol = symbol,
                    .decl = decl,
                });
            }
        }
        return;
    }

    for (0..@intCast(num_args)) |arg_idx| {
        try collectAstSymbolDecls(ctx, z3.Z3_get_app_arg(ctx, app, @intCast(arg_idx)), symbols);
    }
}

const BuiltSmtlibQuery = struct {
    smtlib_z: [:0]const u8,
    decl_symbols: []const z3.Z3_symbol,
    decls: []const z3.Z3_func_decl,
};

fn buildSmtlibForConstraints(
    allocator: std.mem.Allocator,
    solver: *Solver,
    constraints: []const z3.Z3_ast,
) !BuiltSmtlibQuery {
    const ctx = solver.context.ctx;
    var out: std.ArrayList(u8) = .{};
    defer out.deinit(allocator);
    var symbols = ManagedArrayList(SmtSymbolDecl).init(allocator);
    defer symbols.deinit();

    try out.appendSlice(allocator, "(set-logic ALL)\n");
    for (constraints) |constraint| {
        try collectAstSymbolDecls(ctx, constraint, &symbols);
    }
    for (constraints) |constraint| {
        const raw = z3.Z3_ast_to_string(ctx, constraint);
        const text = if (raw == null) "true" else std.mem.span(raw);
        try out.appendSlice(allocator, "(assert ");
        try out.appendSlice(allocator, text);
        try out.appendSlice(allocator, ")\n");
    }
    try out.appendSlice(allocator, "(check-sat)\n");
    const smtlib_z = try out.toOwnedSliceSentinel(allocator, 0);
    const decl_symbols = if (symbols.items.len == 0) &.{} else blk: {
        const owned = try allocator.alloc(z3.Z3_symbol, symbols.items.len);
        for (symbols.items, 0..) |symbol, idx| owned[idx] = symbol.symbol;
        break :blk owned;
    };
    const decls = if (symbols.items.len == 0) &.{} else blk: {
        const owned = try allocator.alloc(z3.Z3_func_decl, symbols.items.len);
        for (symbols.items, 0..) |symbol, idx| owned[idx] = symbol.decl;
        break :blk owned;
    };
    return .{
        .smtlib_z = smtlib_z,
        .decl_symbols = decl_symbols,
        .decls = decls,
    };
}

fn parseLocationString(loc: []const u8) struct { file: []const u8, line: u32, column: u32 } {
    var text = std.mem.trim(u8, loc, " \t\n\r");
    if (std.mem.startsWith(u8, text, "loc(")) {
        text = text[4..];
        if (text.len > 0 and text[text.len - 1] == ')') {
            text = text[0 .. text.len - 1];
        }
    }

    const last_quote = std.mem.lastIndexOfScalar(u8, text, '"') orelse return .{ .file = "", .line = 0, .column = 0 };
    const prev_quote = std.mem.lastIndexOfScalar(u8, text[0..last_quote], '"') orelse return .{ .file = "", .line = 0, .column = 0 };
    const file = text[prev_quote + 1 .. last_quote];

    if (last_quote + 1 >= text.len or text[last_quote + 1] != ':')
        return .{ .file = "", .line = 0, .column = 0 };
    const tail = text[last_quote + 2 ..];
    const sep = std.mem.indexOfScalar(u8, tail, ':') orelse return .{ .file = "", .line = 0, .column = 0 };

    const line_tail = tail[0..sep];
    const col_tail = tail[sep + 1 ..];

    const line_end = std.mem.indexOfNone(u8, line_tail, "0123456789") orelse line_tail.len;
    const col_end = std.mem.indexOfNone(u8, col_tail, "0123456789") orelse col_tail.len;

    const line_str = line_tail[0..line_end];
    const col_str = col_tail[0..col_end];

    const line = std.fmt.parseInt(u32, line_str, 10) catch 0;
    const column = std.fmt.parseInt(u32, col_str, 10) catch 0;
    return .{ .file = file, .line = line, .column = column };
}

const testing = std.testing;

fn testStringRef(comptime s: []const u8) mlir.MlirStringRef {
    return mlir.oraStringRefCreate(s.ptr, s.len);
}

fn testNamedAttr(ctx: mlir.MlirContext, comptime name: []const u8, attr: mlir.MlirAttribute) mlir.MlirNamedAttribute {
    const id = mlir.oraIdentifierGet(ctx, mlir.oraStringRefCreate(name.ptr, name.len));
    return mlir.oraNamedAttributeGet(id, attr);
}

fn testLoadAllDialects(ctx: mlir.MlirContext) void {
    const registry = mlir.oraDialectRegistryCreate();
    defer mlir.oraDialectRegistryDestroy(registry);
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
}

fn buildForInvariantModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("for_invariant_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c4_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 4);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c4_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c4_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(func_body, c0_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c4_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c1_op);

    const lb = mlir.oraOperationGetResult(c0_op, 0);
    const ub = mlir.oraOperationGetResult(c4_op, 0);
    const step = mlir.oraOperationGetResult(c1_op, 0);
    const empty_init_args = [_]mlir.MlirValue{};
    const for_op = mlir.oraScfForOpCreate(mlir_ctx, loc, lb, ub, step, &empty_init_args, empty_init_args.len, false);
    mlir.oraBlockAppendOwnedOperation(func_body, for_op);

    const for_body = mlir.oraScfForOpGetBodyBlock(for_op);
    const for_term = mlir.oraBlockGetTerminator(for_body);
    const inv_cond_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const inv_cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, inv_cond_attr);
    const inv_cond = mlir.oraOperationGetResult(inv_cond_op, 0);
    const inv_op = mlir.oraInvariantOpCreate(mlir_ctx, loc, inv_cond);
    if (mlir.oraOperationIsNull(for_term)) {
        mlir.oraBlockAppendOwnedOperation(for_body, inv_cond_op);
        mlir.oraBlockAppendOwnedOperation(for_body, inv_op);
    } else {
        mlir.oraBlockInsertOwnedOperationBefore(for_body, inv_cond_op, for_term);
        mlir.oraBlockInsertOwnedOperationBefore(for_body, inv_op, for_term);
    }

    const ens_cond_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const ens_cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, ens_cond_attr);
    const ens_cond = mlir.oraOperationGetResult(ens_cond_op, 0);
    const ensures_op = mlir.oraEnsuresOpCreate(mlir_ctx, loc, ens_cond);
    mlir.oraBlockAppendOwnedOperation(func_body, ens_cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ensures_op);

    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildForInvariantConjunctionModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("for_invariant_conjunction_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const param_types = [_]mlir.MlirType{ i1_ty, i1_ty };
    const param_locs = [_]mlir.MlirLocation{ loc, loc };
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c4_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 4);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c4_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c4_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(func_body, c0_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c4_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c1_op);

    const lb = mlir.oraOperationGetResult(c0_op, 0);
    const ub = mlir.oraOperationGetResult(c4_op, 0);
    const step = mlir.oraOperationGetResult(c1_op, 0);
    const empty_init_args = [_]mlir.MlirValue{};
    const for_op = mlir.oraScfForOpCreate(mlir_ctx, loc, lb, ub, step, &empty_init_args, empty_init_args.len, false);
    mlir.oraBlockAppendOwnedOperation(func_body, for_op);

    const a = mlir.oraBlockGetArgument(func_body, 0);
    const b = mlir.oraBlockGetArgument(func_body, 1);

    const for_body = mlir.oraScfForOpGetBodyBlock(for_op);
    const for_term = mlir.oraBlockGetTerminator(for_body);
    const inv_a = mlir.oraInvariantOpCreate(mlir_ctx, loc, a);
    const inv_b = mlir.oraInvariantOpCreate(mlir_ctx, loc, b);
    if (mlir.oraOperationIsNull(for_term)) {
        mlir.oraBlockAppendOwnedOperation(for_body, inv_a);
        mlir.oraBlockAppendOwnedOperation(for_body, inv_b);
    } else {
        mlir.oraBlockInsertOwnedOperationBefore(for_body, inv_a, for_term);
        mlir.oraBlockInsertOwnedOperationBefore(for_body, inv_b, for_term);
    }

    const ensure_cond_op = mlir.oraArithAndIOpCreate(mlir_ctx, loc, a, b);
    const ensure_cond = mlir.oraOperationGetResult(ensure_cond_op, 0);
    const ensures_op = mlir.oraEnsuresOpCreate(mlir_ctx, loc, ensure_cond);
    mlir.oraBlockAppendOwnedOperation(func_body, ensure_cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ensures_op);

    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildForInvariantFailingEnsureModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("for_invariant_failing_ensure_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c4_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 4);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c4_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c4_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(func_body, c0_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c4_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c1_op);
    mlir.oraBlockAppendOwnedOperation(func_body, true_op);
    mlir.oraBlockAppendOwnedOperation(func_body, false_op);

    const lb = mlir.oraOperationGetResult(c0_op, 0);
    const ub = mlir.oraOperationGetResult(c4_op, 0);
    const step = mlir.oraOperationGetResult(c1_op, 0);
    const empty_init_args = [_]mlir.MlirValue{};
    const for_op = mlir.oraScfForOpCreate(mlir_ctx, loc, lb, ub, step, &empty_init_args, empty_init_args.len, false);
    mlir.oraBlockAppendOwnedOperation(func_body, for_op);

    const for_body = mlir.oraScfForOpGetBodyBlock(for_op);
    const for_term = mlir.oraBlockGetTerminator(for_body);
    const inv_op = mlir.oraInvariantOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(true_op, 0));
    if (mlir.oraOperationIsNull(for_term)) {
        mlir.oraBlockAppendOwnedOperation(for_body, inv_op);
    } else {
        mlir.oraBlockInsertOwnedOperationBefore(for_body, inv_op, for_term);
    }

    const ensures_op = mlir.oraEnsuresOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(false_op, 0));
    mlir.oraBlockAppendOwnedOperation(func_body, ensures_op);

    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildForContractInvariantModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("for_contract_invariant_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c5_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 5);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c100_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 100);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c5_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c5_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const c100_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, c100_attr);
    mlir.oraBlockAppendOwnedOperation(func_body, c0_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c5_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c1_op);
    mlir.oraBlockAppendOwnedOperation(func_body, c100_op);

    const lb = mlir.oraOperationGetResult(c0_op, 0);
    const ub = mlir.oraOperationGetResult(c5_op, 0);
    const step = mlir.oraOperationGetResult(c1_op, 0);
    const empty_init_args = [_]mlir.MlirValue{};
    const for_op = mlir.oraScfForOpCreate(mlir_ctx, loc, lb, ub, step, &empty_init_args, empty_init_args.len, false);
    mlir.oraBlockAppendOwnedOperation(func_body, for_op);

    const for_body = mlir.oraScfForOpGetBodyBlock(for_op);
    const for_term = mlir.oraBlockGetTerminator(for_body);
    const iv = mlir.oraBlockGetArgument(for_body, 0);
    const iv_cast_op = mlir.oraArithIndexCastUIOpCreate(mlir_ctx, loc, iv, i256_ty);
    const iv_i256 = mlir.oraOperationGetResult(iv_cast_op, 0);
    const mul_op = mlir.oraArithMulIOpCreate(mlir_ctx, loc, iv_i256, iv_i256);
    const mul = mlir.oraOperationGetResult(mul_op, 0);
    const c100 = mlir.oraOperationGetResult(c100_op, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 7, mul, c100); // ule
    const cmp = mlir.oraOperationGetResult(cmp_op, 0);
    const assert_op = mlir.oraCfAssertOpCreate(mlir_ctx, loc, cmp, testStringRef("mul bounded in loop"));

    if (mlir.oraOperationIsNull(for_term)) {
        mlir.oraBlockAppendOwnedOperation(for_body, iv_cast_op);
        mlir.oraBlockAppendOwnedOperation(for_body, mul_op);
        mlir.oraBlockAppendOwnedOperation(for_body, cmp_op);
        mlir.oraBlockAppendOwnedOperation(for_body, assert_op);
    } else {
        mlir.oraBlockInsertOwnedOperationBefore(for_body, iv_cast_op, for_term);
        mlir.oraBlockInsertOwnedOperationBefore(for_body, mul_op, for_term);
        mlir.oraBlockInsertOwnedOperationBefore(for_body, cmp_op, for_term);
        mlir.oraBlockInsertOwnedOperationBefore(for_body, assert_op, for_term);
    }

    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildConditionalReturnContractInvariantModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("conditional_return_contract_invariant_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const param_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const param_locs = [_]mlir.MlirLocation{ loc, loc };
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const lhs = mlir.oraBlockGetArgument(func_body, 0);
    const rhs = mlir.oraBlockGetArgument(func_body, 1);

    const cond_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, lhs, rhs); // uge
    const cond = mlir.oraOperationGetResult(cond_op, 0);

    const empty_vals = [_]mlir.MlirValue{};
    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, cond);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const then_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, lhs, rhs); // uge
    const then_cond = mlir.oraOperationGetResult(then_cmp_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, then_cond, testStringRef("must hold under conditional return path"));
    const then_yield = mlir.oraYieldOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_cmp_op);
    mlir.oraBlockAppendOwnedOperation(then_block, assert_op);
    mlir.oraBlockAppendOwnedOperation(then_block, then_yield);

    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(func_body, cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildConditionalReturnFallthroughModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("conditional_return_fallthrough_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const param_types = [_]mlir.MlirType{i1_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const flag = mlir.oraBlockGetArgument(func_body, 0);

    const empty_vals = [_]mlir.MlirValue{};
    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, flag);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_then_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, false_then_op);
    mlir.oraBlockAppendOwnedOperation(then_block, then_ret);

    const false_outer_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_outer_val = mlir.oraOperationGetResult(false_outer_op, 0);
    const not_flag_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, flag, false_outer_val); // eq flag false
    const not_flag = mlir.oraOperationGetResult(not_flag_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, not_flag, testStringRef("fallthrough must imply not flag"));
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, false_outer_op);
    mlir.oraBlockAppendOwnedOperation(func_body, not_flag_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildDoubleConditionalReturnFallthroughModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("double_conditional_return_fallthrough_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const param_types = [_]mlir.MlirType{ i1_ty, i1_ty };
    const param_locs = [_]mlir.MlirLocation{ loc, loc };
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const flag0 = mlir.oraBlockGetArgument(func_body, 0);
    const flag1 = mlir.oraBlockGetArgument(func_body, 1);

    const empty_vals = [_]mlir.MlirValue{};
    const first_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, flag0);
    const first_then = mlir.oraConditionalReturnOpGetThenBlock(first_ret);
    const first_then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(first_then, first_then_ret);

    const second_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, flag1);
    const second_then = mlir.oraConditionalReturnOpGetThenBlock(second_ret);
    const second_then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(second_then, second_then_ret);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false0 = mlir.oraOperationGetResult(false0_op, 0);
    const false1 = mlir.oraOperationGetResult(false1_op, 0);
    const not_flag0_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, flag0, false0);
    const not_flag1_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, flag1, false1);
    const not_flag0 = mlir.oraOperationGetResult(not_flag0_op, 0);
    const not_flag1 = mlir.oraOperationGetResult(not_flag1_op, 0);
    const both_ok_op = mlir.oraArithAndIOpCreate(mlir_ctx, loc, not_flag0, not_flag1);
    const both_ok = mlir.oraOperationGetResult(both_ok_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, both_ok, testStringRef("fallthrough must imply both flags are false"));
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, first_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, second_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, false0_op);
    mlir.oraBlockAppendOwnedOperation(func_body, false1_op);
    mlir.oraBlockAppendOwnedOperation(func_body, not_flag0_op);
    mlir.oraBlockAppendOwnedOperation(func_body, not_flag1_op);
    mlir.oraBlockAppendOwnedOperation(func_body, both_ok_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildConditionalReturnStatefulDivModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("conditional_return_stateful_div_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"))),
    };
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const param_types = [_]mlir.MlirType{i256_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const divisor = mlir.oraBlockGetArgument(func_body, 0);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const is_zero_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, divisor, zero); // eq
    const is_zero = mlir.oraOperationGetResult(is_zero_op, 0);

    const empty_vals = [_]mlir.MlirValue{};
    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, is_zero);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_ret);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one = mlir.oraOperationGetResult(one_op, 0);
    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, one, divisor);
    const div_val = mlir.oraOperationGetResult(div_op, 0);
    const store_op = mlir.oraSStoreOpCreate(mlir_ctx, loc, div_val, testStringRef("counter"));
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, is_zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, one_op);
    mlir.oraBlockAppendOwnedOperation(func_body, div_op);
    mlir.oraBlockAppendOwnedOperation(func_body, store_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildConditionalReturnTStoreDivModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("conditional_return_tstore_div_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"))),
    };
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const param_types = [_]mlir.MlirType{i256_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const divisor = mlir.oraBlockGetArgument(func_body, 0);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const is_zero_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, divisor, zero); // eq
    const is_zero = mlir.oraOperationGetResult(is_zero_op, 0);

    const empty_vals = [_]mlir.MlirValue{};
    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, is_zero);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_ret);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one = mlir.oraOperationGetResult(one_op, 0);
    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, one, divisor);
    const div_val = mlir.oraOperationGetResult(div_op, 0);
    const tstore_op = mlir.oraTStoreOpCreate(mlir_ctx, loc, div_val, testStringRef("pending"));
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, is_zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, one_op);
    mlir.oraBlockAppendOwnedOperation(func_body, div_op);
    mlir.oraBlockAppendOwnedOperation(func_body, tstore_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildConditionalReturnMapStoreDivModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("conditional_return_map_store_div_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"))),
    };
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const param_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const param_locs = [_]mlir.MlirLocation{ loc, loc };
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const divisor = mlir.oraBlockGetArgument(func_body, 0);
    const key = mlir.oraBlockGetArgument(func_body, 1);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const is_zero_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, divisor, zero); // eq
    const is_zero = mlir.oraOperationGetResult(is_zero_op, 0);

    const empty_vals = [_]mlir.MlirValue{};
    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, is_zero);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_ret);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one = mlir.oraOperationGetResult(one_op, 0);
    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, one, divisor);
    const div_val = mlir.oraOperationGetResult(div_op, 0);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);
    const map_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, testStringRef("balances"), map_ty);
    const map_store = mlir.oraMapStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(map_load, 0), key, div_val);
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, is_zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, one_op);
    mlir.oraBlockAppendOwnedOperation(func_body, div_op);
    mlir.oraBlockAppendOwnedOperation(func_body, map_load);
    mlir.oraBlockAppendOwnedOperation(func_body, map_store);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildConditionalReturnMapGetDivModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("conditional_return_map_get_div_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"))),
    };
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);
    const param_types = [_]mlir.MlirType{i256_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const key = mlir.oraBlockGetArgument(func_body, 0);

    const borrows_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, testStringRef("borrows"), map_ty);
    const borrow_val_op = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(borrows_load, 0), key, i256_ty);
    const borrow_val = mlir.oraOperationGetResult(borrow_val_op, 0);
    const collateral_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, testStringRef("collateral"), map_ty);
    const collateral_val_op = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(collateral_load, 0), key, i256_ty);
    const collateral_val = mlir.oraOperationGetResult(collateral_val_op, 0);
    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const is_zero_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, borrow_val, zero); // eq
    const is_zero = mlir.oraOperationGetResult(is_zero_op, 0);
    const collateral_zero_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, collateral_val, zero); // eq
    const collateral_zero = mlir.oraOperationGetResult(collateral_zero_op, 0);

    const empty_vals = [_]mlir.MlirValue{};
    const first_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, is_zero);
    const first_then = mlir.oraConditionalReturnOpGetThenBlock(first_ret);
    const first_then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(first_then, first_then_ret);

    const second_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, collateral_zero);
    const second_then = mlir.oraConditionalReturnOpGetThenBlock(second_ret);
    const second_then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(second_then, second_then_ret);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one = mlir.oraOperationGetResult(one_op, 0);
    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, one, borrow_val);
    const div_val = mlir.oraOperationGetResult(div_op, 0);
    const health_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, testStringRef("health_factors"), map_ty);
    const map_store = mlir.oraMapStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(health_load, 0), key, div_val);
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, borrows_load);
    mlir.oraBlockAppendOwnedOperation(func_body, borrow_val_op);
    mlir.oraBlockAppendOwnedOperation(func_body, collateral_load);
    mlir.oraBlockAppendOwnedOperation(func_body, collateral_val_op);
    mlir.oraBlockAppendOwnedOperation(func_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, is_zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, collateral_zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, first_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, second_ret);
    mlir.oraBlockAppendOwnedOperation(func_body, one_op);
    mlir.oraBlockAppendOwnedOperation(func_body, div_op);
    mlir.oraBlockAppendOwnedOperation(func_body, health_load);
    mlir.oraBlockAppendOwnedOperation(func_body, map_store);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildUntaggedCfAssertModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("cf_assert_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const cond_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, cond_attr);
    const cond = mlir.oraOperationGetResult(cond_op, 0);
    const assert_op = mlir.oraCfAssertOpCreate(mlir_ctx, loc, cond, testStringRef("assert"));
    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildUntaggedOraAssertModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("ora_assert_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const cond_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, cond_attr);
    const cond = mlir.oraOperationGetResult(cond_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, cond, testStringRef("assert"));
    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildGuardOraAssertModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard_assert_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const cond_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, cond_attr);
    const cond = mlir.oraOperationGetResult(cond_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, cond, testStringRef("guard violation path: cond"));
    mlir.oraOperationSetAttributeByName(assert_op, testStringRef("ora.verification_type"), mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard")));
    mlir.oraOperationSetAttributeByName(assert_op, testStringRef("ora.verification_context"), mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard_clause")));
    mlir.oraOperationSetAttributeByName(assert_op, testStringRef("ora.guard_id"), mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard:test:clause")));
    const assume_op = mlir.oraAssumeOpCreate(mlir_ctx, loc, cond);
    mlir.oraOperationSetAttributeByName(assume_op, testStringRef("ora.verification_context"), mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard_clause")));
    mlir.oraOperationSetAttributeByName(assume_op, testStringRef("ora.guard_id"), mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard:test:clause")));
    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assume_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildCheckedMulOraAssertModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("checked_mul_assert_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const param_types = [_]mlir.MlirType{i256_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, 1);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const lhs = mlir.oraBlockGetArgument(func_body, 0);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10000);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);
    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_const, 0);
    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const bool_true = mlir.oraOperationGetResult(true_const, 0);

    const mul_op = mlir.oraArithMulIOpCreate(mlir_ctx, loc, lhs, rhs);
    const mul = mlir.oraOperationGetResult(mul_op, 0);
    const rhs_non_zero_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 1, rhs, zero); // ne
    const rhs_non_zero = mlir.oraOperationGetResult(rhs_non_zero_op, 0);
    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, mul, rhs);
    mlir.oraOperationSetAttributeByName(div_op, testStringRef("ora.guard_internal"), mlir.oraBoolAttrCreate(mlir_ctx, true));
    const recovered = mlir.oraOperationGetResult(div_op, 0);
    const overflow_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 1, recovered, lhs); // ne
    const overflow_cmp = mlir.oraOperationGetResult(overflow_cmp_op, 0);
    const and_op = mlir.oraArithAndIOpCreate(mlir_ctx, loc, overflow_cmp, rhs_non_zero);
    const overflow_flag = mlir.oraOperationGetResult(and_op, 0);
    const not_overflow_op = mlir.oraArithXorIOpCreate(mlir_ctx, loc, overflow_flag, bool_true);
    const not_overflow = mlir.oraOperationGetResult(not_overflow_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, not_overflow, testStringRef("checked multiplication overflow"));

    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, rhs_const);
    mlir.oraBlockAppendOwnedOperation(func_body, zero_const);
    mlir.oraBlockAppendOwnedOperation(func_body, true_const);
    mlir.oraBlockAppendOwnedOperation(func_body, mul_op);
    mlir.oraBlockAppendOwnedOperation(func_body, rhs_non_zero_op);
    mlir.oraBlockAppendOwnedOperation(func_body, div_op);
    mlir.oraBlockAppendOwnedOperation(func_body, overflow_cmp_op);
    mlir.oraBlockAppendOwnedOperation(func_body, and_op);
    mlir.oraBlockAppendOwnedOperation(func_body, not_overflow_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn findFirstOpByNameInBlock(block: mlir.MlirBlock, name: []const u8) ?mlir.MlirOperation {
    var op = mlir.oraBlockGetFirstOperation(block);
    while (!mlir.oraOperationIsNull(op)) : (op = mlir.oraOperationGetNextInBlock(op)) {
        const name_ref = mlir.oraOperationGetName(op);
        defer @import("mlir_c_api").freeStringRef(name_ref);
        if (name_ref.data != null and std.mem.eql(u8, name_ref.data[0..name_ref.length], name)) {
            return op;
        }
    }
    return null;
}

fn buildPublicCallsPrivateAssertModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const empty_vals = [_]mlir.MlirValue{};

    const private_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private_helper"));
    const private_visibility_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private"));
    const private_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", private_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", private_visibility_attr),
    };
    const private_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &private_attrs, private_attrs.len, &empty_types, &empty_locs, 0);
    const private_body = mlir.oraFuncOpGetBodyBlock(private_func_op);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, false_val, testStringRef("private assert should flow to caller"));
    const private_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(private_body, false_op);
    mlir.oraBlockAppendOwnedOperation(private_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(private_body, private_ret);

    const public_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public_entry"));
    const public_visibility_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"));
    const public_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", public_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", public_visibility_attr),
    };
    const public_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &public_attrs, public_attrs.len, &empty_types, &empty_locs, 0);
    const public_body = mlir.oraFuncOpGetBodyBlock(public_func_op);

    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        testStringRef("private_helper"),
        &empty_vals,
        empty_vals.len,
        &empty_types,
        empty_types.len,
    );
    const public_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(public_body, call_op);
    mlir.oraBlockAppendOwnedOperation(public_body, public_ret);

    mlir.oraBlockAppendOwnedOperation(module_body, public_func_op);
    mlir.oraBlockAppendOwnedOperation(module_body, private_func_op);
    return module;
}

fn buildPublicCallsPrivateAssertReturningModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const one_type = [_]mlir.MlirType{i256_ty};
    const one_loc = [_]mlir.MlirLocation{loc};
    const empty_vals = [_]mlir.MlirValue{};

    const private_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private_helper_ret"))),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private"))),
    };
    const private_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &private_attrs, private_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const private_body = mlir.oraFuncOpGetBodyBlock(private_func_op);

    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0));
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7));
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(false_op, 0), testStringRef("private result assert should flow to caller"));
    const private_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)}, 1);

    mlir.oraBlockAppendOwnedOperation(private_body, false_op);
    mlir.oraBlockAppendOwnedOperation(private_body, seven_op);
    mlir.oraBlockAppendOwnedOperation(private_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(private_body, private_ret);

    const public_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public_entry"))),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"))),
    };
    const public_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &public_attrs, public_attrs.len, &one_type, &one_loc, one_type.len);
    const public_body = mlir.oraFuncOpGetBodyBlock(public_func_op);

    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        testStringRef("private_helper_ret"),
        &empty_vals,
        empty_vals.len,
        &one_type,
        one_type.len,
    );
    const public_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(call_op, 0)}, 1);

    mlir.oraBlockAppendOwnedOperation(public_body, call_op);
    mlir.oraBlockAppendOwnedOperation(public_body, public_ret);

    mlir.oraBlockAppendOwnedOperation(module_body, public_func_op);
    mlir.oraBlockAppendOwnedOperation(module_body, private_func_op);
    return module;
}

fn buildPublicCallsPrivateCheckedSubGuardedModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const arg_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const arg_locs = [_]mlir.MlirLocation{ loc, loc };
    const one_ret_type = [_]mlir.MlirType{i256_ty};
    const empty_vals = [_]mlir.MlirValue{};

    const private_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private_checked_sub"))),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private"))),
    };
    const private_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &private_attrs, private_attrs.len, &arg_types, &arg_locs, arg_types.len);
    const private_body = mlir.oraFuncOpGetBodyBlock(private_func_op);
    const lhs = mlir.oraBlockGetArgument(private_body, 0);
    const rhs = mlir.oraBlockGetArgument(private_body, 1);

    const sub_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, lhs, rhs);
    const sub_val = mlir.oraOperationGetResult(sub_op, 0);
    const underflow_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, lhs, rhs); // ult
    const underflow = mlir.oraOperationGetResult(underflow_op, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1));
    const ok_op = mlir.oraArithXorIOpCreate(mlir_ctx, loc, underflow, mlir.oraOperationGetResult(true_op, 0));
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(ok_op, 0), testStringRef("checked subtraction overflow"));
    const private_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{sub_val}, 1);

    mlir.oraBlockAppendOwnedOperation(private_body, sub_op);
    mlir.oraBlockAppendOwnedOperation(private_body, underflow_op);
    mlir.oraBlockAppendOwnedOperation(private_body, true_op);
    mlir.oraBlockAppendOwnedOperation(private_body, ok_op);
    mlir.oraBlockAppendOwnedOperation(private_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(private_body, private_ret);

    const public_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public_entry"))),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"))),
    };
    const public_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &public_attrs, public_attrs.len, &arg_types, &arg_locs, arg_types.len);
    const public_body = mlir.oraFuncOpGetBodyBlock(public_func_op);
    const pub_lhs = mlir.oraBlockGetArgument(public_body, 0);
    const pub_rhs = mlir.oraBlockGetArgument(public_body, 1);

    const needs_return_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, pub_lhs, pub_rhs); // ult
    const needs_return = mlir.oraOperationGetResult(needs_return_op, 0);
    const cond_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, needs_return);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(cond_ret);
    const then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_ret);

    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        testStringRef("private_checked_sub"),
        &[_]mlir.MlirValue{ pub_lhs, pub_rhs },
        2,
        &one_ret_type,
        one_ret_type.len,
    );
    const public_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(public_body, needs_return_op);
    mlir.oraBlockAppendOwnedOperation(public_body, cond_ret);
    mlir.oraBlockAppendOwnedOperation(public_body, call_op);
    mlir.oraBlockAppendOwnedOperation(public_body, public_ret);

    mlir.oraBlockAppendOwnedOperation(module_body, public_func_op);
    mlir.oraBlockAppendOwnedOperation(module_body, private_func_op);
    return module;
}

fn buildPublicCallsPrivateRequiresGuardedAssertModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const one_type = [_]mlir.MlirType{i256_ty};
    const one_loc = [_]mlir.MlirLocation{loc};
    const empty_types = [_]mlir.MlirType{};
    const empty_vals = [_]mlir.MlirValue{};

    const private_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private_helper"));
    const private_visibility_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private"));
    const private_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", private_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", private_visibility_attr),
    };
    const private_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &private_attrs, private_attrs.len, &one_type, &one_loc, one_type.len);
    const private_body = mlir.oraFuncOpGetBodyBlock(private_func_op);
    const private_arg = mlir.oraBlockGetArgument(private_body, 0);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one_val = mlir.oraOperationGetResult(one_op, 0);

    const max_minus_one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, -2);
    const max_minus_one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, max_minus_one_attr);
    const max_minus_one_val = mlir.oraOperationGetResult(max_minus_one_op, 0);

    const precond_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 7, private_arg, max_minus_one_val); // ule
    const precond_cmp_val = mlir.oraOperationGetResult(precond_cmp_op, 0);

    const precond_assert = mlir.oraCfAssertOpCreate(mlir_ctx, loc, precond_cmp_val, testStringRef("private precondition"));
    const requires_attr = mlir.oraBoolAttrCreate(mlir_ctx, true);
    mlir.oraOperationSetAttributeByName(precond_assert, testStringRef("ora.requires"), requires_attr);

    const add_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, private_arg, one_val);
    const add_val = mlir.oraOperationGetResult(add_op, 0);
    const overflow_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, add_val, private_arg); // uge
    const overflow_cmp_val = mlir.oraOperationGetResult(overflow_cmp_op, 0);
    const body_assert = mlir.oraAssertOpCreate(mlir_ctx, loc, overflow_cmp_val, testStringRef("private assert should be guarded by requires"));
    const private_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(private_body, one_op);
    mlir.oraBlockAppendOwnedOperation(private_body, max_minus_one_op);
    mlir.oraBlockAppendOwnedOperation(private_body, precond_cmp_op);
    mlir.oraBlockAppendOwnedOperation(private_body, precond_assert);
    mlir.oraBlockAppendOwnedOperation(private_body, add_op);
    mlir.oraBlockAppendOwnedOperation(private_body, overflow_cmp_op);
    mlir.oraBlockAppendOwnedOperation(private_body, body_assert);
    mlir.oraBlockAppendOwnedOperation(private_body, private_ret);

    const public_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public_entry"));
    const public_visibility_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"));
    const public_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", public_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", public_visibility_attr),
    };
    const public_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &public_attrs, public_attrs.len, &one_type, &one_loc, one_type.len);
    const public_body = mlir.oraFuncOpGetBodyBlock(public_func_op);
    const public_arg = mlir.oraBlockGetArgument(public_body, 0);
    const call_args = [_]mlir.MlirValue{public_arg};

    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        testStringRef("private_helper"),
        &call_args,
        call_args.len,
        &empty_types,
        empty_types.len,
    );
    const public_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(public_body, call_op);
    mlir.oraBlockAppendOwnedOperation(public_body, public_ret);

    mlir.oraBlockAppendOwnedOperation(module_body, public_func_op);
    mlir.oraBlockAppendOwnedOperation(module_body, private_func_op);
    return module;
}

fn buildPublicCallsPrivateRequiresGuardedAssertReturningModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const one_type = [_]mlir.MlirType{i256_ty};
    const one_loc = [_]mlir.MlirLocation{loc};

    const private_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private_helper_ret"))),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("private"))),
    };
    const private_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &private_attrs, private_attrs.len, &one_type, &one_loc, one_type.len);
    const private_body = mlir.oraFuncOpGetBodyBlock(private_func_op);
    const private_arg = mlir.oraBlockGetArgument(private_body, 0);

    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1));
    const max_minus_one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, -2));
    const precond_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 7, private_arg, mlir.oraOperationGetResult(max_minus_one_op, 0)); // ule
    const precond_assert = mlir.oraCfAssertOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(precond_cmp_op, 0), testStringRef("private result precondition"));
    mlir.oraOperationSetAttributeByName(precond_assert, testStringRef("ora.requires"), mlir.oraBoolAttrCreate(mlir_ctx, true));

    const add_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, private_arg, mlir.oraOperationGetResult(one_op, 0));
    const overflow_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, mlir.oraOperationGetResult(add_op, 0), private_arg); // uge
    const body_assert = mlir.oraAssertOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(overflow_cmp_op, 0), testStringRef("private result assert should be guarded by requires"));
    const private_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(add_op, 0)}, 1);

    mlir.oraBlockAppendOwnedOperation(private_body, one_op);
    mlir.oraBlockAppendOwnedOperation(private_body, max_minus_one_op);
    mlir.oraBlockAppendOwnedOperation(private_body, precond_cmp_op);
    mlir.oraBlockAppendOwnedOperation(private_body, precond_assert);
    mlir.oraBlockAppendOwnedOperation(private_body, add_op);
    mlir.oraBlockAppendOwnedOperation(private_body, overflow_cmp_op);
    mlir.oraBlockAppendOwnedOperation(private_body, body_assert);
    mlir.oraBlockAppendOwnedOperation(private_body, private_ret);

    const public_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public_entry"))),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"))),
    };
    const public_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &public_attrs, public_attrs.len, &one_type, &one_loc, one_type.len);
    const public_body = mlir.oraFuncOpGetBodyBlock(public_func_op);
    const public_arg = mlir.oraBlockGetArgument(public_body, 0);
    const call_args = [_]mlir.MlirValue{public_arg};

    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        testStringRef("private_helper_ret"),
        &call_args,
        call_args.len,
        &one_type,
        one_type.len,
    );
    const public_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(call_op, 0)}, 1);

    mlir.oraBlockAppendOwnedOperation(public_body, call_op);
    mlir.oraBlockAppendOwnedOperation(public_body, public_ret);

    mlir.oraBlockAppendOwnedOperation(module_body, public_func_op);
    mlir.oraBlockAppendOwnedOperation(module_body, private_func_op);
    return module;
}

fn buildPublicCallsPublicAssertModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const empty_vals = [_]mlir.MlirValue{};

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public_helper"));
    const helper_visibility_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"));
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", helper_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", helper_visibility_attr),
    };
    const helper_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &empty_types, &empty_locs, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func_op);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, false_val, testStringRef("public helper assert"));
    const helper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(helper_body, false_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, assert_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_ret);

    const caller_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public_entry"));
    const caller_visibility_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"));
    const caller_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", caller_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", caller_visibility_attr),
    };
    const caller_func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &caller_attrs, caller_attrs.len, &empty_types, &empty_locs, 0);
    const caller_body = mlir.oraFuncOpGetBodyBlock(caller_func_op);

    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        testStringRef("public_helper"),
        &empty_vals,
        empty_vals.len,
        &empty_types,
        empty_types.len,
    );
    const caller_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);

    mlir.oraBlockAppendOwnedOperation(caller_body, call_op);
    mlir.oraBlockAppendOwnedOperation(caller_body, caller_ret);

    mlir.oraBlockAppendOwnedOperation(module_body, caller_func_op);
    mlir.oraBlockAppendOwnedOperation(module_body, helper_func_op);
    return module;
}

fn buildPathAssumeEnsuresModule(mlir_ctx: mlir.MlirContext, path_assume_value: i64, ensures_value: i64) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("path_assume_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const assume_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, path_assume_value);
    const assume_cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, assume_attr);
    const assume_cond = mlir.oraOperationGetResult(assume_cond_op, 0);
    const assume_op = mlir.oraAssumeOpCreate(mlir_ctx, loc, assume_cond);
    const origin_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("path"));
    mlir.oraOperationSetAttributeByName(assume_op, testStringRef("ora.assume_origin"), origin_attr);

    const ensure_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, ensures_value);
    const ensure_cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, ensure_attr);
    const ensure_cond = mlir.oraOperationGetResult(ensure_cond_op, 0);
    const ensures_op = mlir.oraEnsuresOpCreate(mlir_ctx, loc, ensure_cond);

    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);

    mlir.oraBlockAppendOwnedOperation(func_body, assume_cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, assume_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ensure_cond_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ensures_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildMalformedAnnotationModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const body = mlir.oraModuleGetBody(module);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("broken_ensures"))),
        testNamedAttr(mlir_ctx, "ora.visibility", mlir.oraStringAttrCreate(mlir_ctx, testStringRef("public"))),
    };
    const func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &attrs, attrs.len, &[_]mlir.MlirType{i256_ty}, &[_]mlir.MlirLocation{loc}, 1);
    const func_body = mlir.oraFuncOpGetBodyBlock(func);
    const arg = mlir.oraBlockGetArgument(func_body, 0);

    const zero = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0),
    );
    const malformed_cmp = mlir.oraCmpOpCreate(mlir_ctx, loc, testStringRef(""), arg, mlir.oraOperationGetResult(zero, 0), i1_ty);
    const ensures = mlir.oraEnsuresOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(malformed_cmp, 0));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{arg}, 1);

    mlir.oraBlockAppendOwnedOperation(func_body, zero);
    mlir.oraBlockAppendOwnedOperation(func_body, malformed_cmp);
    mlir.oraBlockAppendOwnedOperation(func_body, ensures);
    mlir.oraBlockAppendOwnedOperation(func_body, ret);
    mlir.oraBlockAppendOwnedOperation(body, func);
    return module;
}

fn buildGlobalContractInvariantModule(mlir_ctx: mlir.MlirContext, invariant_value: i64) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const inv_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, invariant_value);
    const inv_cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, inv_attr);
    const inv_op = mlir.oraInvariantOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(inv_cond_op, 0));
    mlir.oraBlockAppendOwnedOperation(module_body, inv_cond_op);
    mlir.oraBlockAppendOwnedOperation(module_body, inv_op);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("global_invariant_target"));
    const visibility_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("pub"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
        testNamedAttr(mlir_ctx, "ora.visibility", visibility_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const empty_return_vals = [_]mlir.MlirValue{};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);
    mlir.oraBlockAppendOwnedOperation(module_body, func_op);

    return module;
}

fn buildBranchPathGuardsModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("branch_path_guards_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const param_types = [_]mlir.MlirType{i1_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const flag = mlir.oraBlockGetArgument(func_body, 0);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const true_val = mlir.oraOperationGetResult(true_op, 0);
    const false_val = mlir.oraOperationGetResult(false_op, 0);

    const empty_result_types = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, flag, &empty_result_types, empty_result_types.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const origin_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("path"));
    const then_assume = mlir.oraAssumeOpCreate(mlir_ctx, loc, flag);
    mlir.oraOperationSetAttributeByName(then_assume, testStringRef("ora.assume_origin"), origin_attr);
    const then_guard = mlir.oraRefinementGuardOpCreate(mlir_ctx, loc, true_val, testStringRef("branch guard then"));
    const then_guard_id = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard:test:then"));
    const refinement_kind = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("min_value"));
    mlir.oraOperationSetAttributeByName(then_guard, testStringRef("ora.guard_id"), then_guard_id);
    mlir.oraOperationSetAttributeByName(then_guard, testStringRef("ora.refinement_kind"), refinement_kind);
    const empty_vals = [_]mlir.MlirValue{};
    const then_yield = mlir.oraScfYieldOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_assume);
    mlir.oraBlockAppendOwnedOperation(then_block, then_guard);
    mlir.oraBlockAppendOwnedOperation(then_block, then_yield);

    const else_cmp = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, flag, false_val);
    const else_cond = mlir.oraOperationGetResult(else_cmp, 0);
    const else_assume = mlir.oraAssumeOpCreate(mlir_ctx, loc, else_cond);
    mlir.oraOperationSetAttributeByName(else_assume, testStringRef("ora.assume_origin"), origin_attr);
    const else_guard = mlir.oraRefinementGuardOpCreate(mlir_ctx, loc, true_val, testStringRef("branch guard else"));
    const else_guard_id = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard:test:else"));
    mlir.oraOperationSetAttributeByName(else_guard, testStringRef("ora.guard_id"), else_guard_id);
    mlir.oraOperationSetAttributeByName(else_guard, testStringRef("ora.refinement_kind"), refinement_kind);
    const else_yield = mlir.oraScfYieldOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(else_block, else_cmp);
    mlir.oraBlockAppendOwnedOperation(else_block, else_assume);
    mlir.oraBlockAppendOwnedOperation(else_block, else_guard);
    mlir.oraBlockAppendOwnedOperation(else_block, else_yield);

    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len);
    mlir.oraBlockAppendOwnedOperation(func_body, true_op);
    mlir.oraBlockAppendOwnedOperation(func_body, false_op);
    mlir.oraBlockAppendOwnedOperation(func_body, if_op);
    mlir.oraBlockAppendOwnedOperation(func_body, ret_op);

    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

fn buildLinearPathAssumeGuardModule(mlir_ctx: mlir.MlirContext) mlir.MlirModule {
    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    const module_body = mlir.oraModuleGetBody(module);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, testStringRef("linear_path_guard_test"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        testNamedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const param_types = [_]mlir.MlirType{i1_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const flag = mlir.oraBlockGetArgument(func_body, 0);

    const assume_op = mlir.oraAssumeOpCreate(mlir_ctx, loc, flag);
    mlir.oraOperationSetAttributeByName(
        assume_op,
        testStringRef("ora.assume_origin"),
        mlir.oraStringAttrCreate(mlir_ctx, testStringRef("path")),
    );

    const guard_op = mlir.oraRefinementGuardOpCreate(mlir_ctx, loc, flag, testStringRef("linear path guard"));
    mlir.oraOperationSetAttributeByName(
        guard_op,
        testStringRef("ora.guard_id"),
        mlir.oraStringAttrCreate(mlir_ctx, testStringRef("guard:test:linear")),
    );
    mlir.oraOperationSetAttributeByName(
        guard_op,
        testStringRef("ora.refinement_kind"),
        mlir.oraStringAttrCreate(mlir_ctx, testStringRef("min_value")),
    );

    const empty_vals = [_]mlir.MlirValue{};
    mlir.oraBlockAppendOwnedOperation(func_body, assume_op);
    mlir.oraBlockAppendOwnedOperation(func_body, guard_op);
    mlir.oraBlockAppendOwnedOperation(func_body, mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_vals, empty_vals.len));
    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    return module;
}

test "scf.for loop invariants capture loop exit conditions" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildForInvariantModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);

    var loop_inv_count: usize = 0;
    for (pass.encoded_annotations.items) |ann| {
        if (ann.kind != .LoopInvariant) continue;
        loop_inv_count += 1;
        try testing.expect(ann.loop_step_condition != null);
        try testing.expect(ann.loop_exit_condition != null);
    }
    try testing.expectEqual(@as(usize, 1), loop_inv_count);
}

test "prepared queries include invariant-step for scf.for" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildForInvariantModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var step_count: usize = 0;
    for (queries.items) |q| {
        if (q.kind == .LoopInvariantStep) {
            step_count += 1;
        }
    }
    try testing.expectEqual(@as(usize, 1), step_count);
}

test "annotation extraction failure is reported as unknown verification error" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildMalformedAnnotationModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    const result = try pass.runVerificationPass(module);
    defer {
        var mutable = result;
        mutable.deinit();
    }

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors.items.len);
    try testing.expect(std.mem.containsAtLeast(u8, result.errors.items[0].message, 1, "annotation extraction"));
    try testing.expect(std.mem.containsAtLeast(u8, result.errors.items[0].message, 1, "Unsupported"));
}

test "prepared queries include invariant-post for scf.for" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildForInvariantModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var post_count: usize = 0;
    for (queries.items) |q| {
        if (q.kind == .LoopInvariantPost) {
            post_count += 1;
        }
    }
    try testing.expectEqual(@as(usize, 1), post_count);
}

test "invariant-post query conjoins loop invariants from same loop" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildForInvariantConjunctionModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_post_query = false;
    for (queries.items) |q| {
        if (q.kind != .LoopInvariantPost) continue;
        found_post_query = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_post_query);
}

test "sequential verification continues past failing ensures to check loop-post" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildForInvariantFailingEnsureModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();

    var ensures_errors: usize = 0;
    var loop_post_errors: usize = 0;
    for (result.errors.items) |err| {
        if (std.mem.indexOf(u8, err.message, "failed to prove ensures") != null) {
            ensures_errors += 1;
        }
        if (std.mem.indexOf(u8, err.message, "failed to prove postcondition from loop invariant at loop exit") != null) {
            loop_post_errors += 1;
        }
    }
    try testing.expectEqual(@as(usize, 1), ensures_errors);
    try testing.expectEqual(@as(usize, 1), loop_post_errors);
}

test "global contract invariants attach to real functions instead of unknown" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildGlobalContractInvariantModule(mlir_ctx, 1);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_target_contract_invariant = false;
    for (queries.items) |q| {
        try testing.expect(!std.mem.eql(u8, q.function_name, "unknown"));
        if (std.mem.eql(u8, q.function_name, "global_invariant_target") and
            q.kind == .Obligation and
            q.obligation_kind == .ContractInvariant)
        {
            found_target_contract_invariant = true;
        }
    }
    try testing.expect(found_target_contract_invariant);
}

test "full verify mode treats untagged cf.assert as obligation" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildUntaggedCfAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var saw_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            saw_contract_obligation = true;
            break;
        }
    }
    try testing.expect(saw_contract_obligation);
}

test "basic verify mode ignores untagged cf.assert as obligation" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Basic);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildUntaggedCfAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    for (queries.items) |q| {
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            try testing.expect(false);
        }
    }
}

test "full verify mode treats untagged ora.assert as obligation" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildUntaggedOraAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var saw_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            saw_contract_obligation = true;
            break;
        }
    }
    try testing.expect(saw_contract_obligation);
}

test "full verify mode classifies guard-tagged ora.assert as guard obligation" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildGuardOraAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var saw_guard_obligation = false;
    for (queries.items) |q| {
        if (q.kind == .Obligation and q.obligation_kind == .Guard) {
            saw_guard_obligation = true;
            break;
        }
    }
    try testing.expect(saw_guard_obligation);
}

test "verified guard obligations populate proven guard ids" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildGuardOraAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expect(result.proven_guard_ids.contains("guard:test:clause"));
}

test "full verify mode simplifies checked multiply assert obligations before SMT-LIB emission" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildCheckedMulOraAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        try testing.expect(std.mem.indexOf(u8, q.smtlib_z, "(bvudiv") != null);
        try testing.expect(std.mem.indexOf(u8, q.smtlib_z, "(bvmul") == null);
    }
    try testing.expect(found_contract_obligation);
}

test "checked multiply assert specialization matches verification test module directly" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildCheckedMulOraAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    const module_body = mlir.oraModuleGetBody(module);
    const func_op = findFirstOpByNameInBlock(module_body, "func.func") orelse return error.TestUnexpectedResult;
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);
    const assert_op = findFirstOpByNameInBlock(func_body, "ora.assert") orelse return error.TestUnexpectedResult;

    const specialized = try pass.encoder.tryEncodeAssertCondition(assert_op, .Current);
    try testing.expect(specialized != null);

    const specialized_text = z3.Z3_ast_to_string(pass.encoder.context.ctx, specialized.?);
    try testing.expect(std.mem.indexOf(u8, std.mem.span(specialized_text), "(bvmul") == null);
}

test "basic verify mode ignores untagged ora.assert as obligation" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Basic);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildUntaggedOraAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    for (queries.items) |q| {
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            try testing.expect(false);
        }
    }
}

test "private callee assert is enforced through reachable public call path" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Basic);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPublicCallsPrivateAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var saw_public_obligation = false;
    for (queries.items) |q| {
        if (!std.mem.eql(u8, q.function_name, "public_entry")) continue;
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            saw_public_obligation = true;
            break;
        }
    }
    try testing.expect(saw_public_obligation);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();
    try testing.expect(!result.success);
    try testing.expect(result.errors.items.len > 0);
}

test "private result callee assert is enforced through reachable public call path" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Basic);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPublicCallsPrivateAssertReturningModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| q.deinit(testing.allocator);
        queries.deinit();
    }

    var saw_public_obligation = false;
    for (queries.items) |q| {
        if (!std.mem.eql(u8, q.function_name, "public_entry")) continue;
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            saw_public_obligation = true;
            break;
        }
    }
    try testing.expect(saw_public_obligation);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();
    try testing.expect(!result.success);
    try testing.expect(result.errors.items.len > 0);
}

test "private checked arithmetic callee is guarded by caller fallthrough path" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPublicCallsPrivateCheckedSubGuardedModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "branch merge preserves untouched global base state" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    var base = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer base.deinit(testing.allocator);
    var then_state = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer then_state.deinit(testing.allocator);
    var else_state = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer else_state.deinit(testing.allocator);

    const bool_sort = z3.Z3_mk_bool_sort(pass.encoder.context.ctx);
    const i256_sort = z3.Z3_mk_bv_sort(pass.encoder.context.ctx, 256);
    const cond = try pass.encoder.mkVariable("branch_cond", bool_sort);
    const one = z3.Z3_mk_numeral(pass.encoder.context.ctx, "1", i256_sort);

    try then_state.global_map.put(try testing.allocator.dupe(u8, "counter"), one);

    try pass.mergeEncoderBranchState(cond, &base, &then_state, &else_state);

    const merged = pass.encoder.global_map.get("counter").?;
    const untouched_base = pass.encoder.global_entry_map.get("counter").?;

    var solver = try Solver.init(pass.encoder.context, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(pass.encoder.context.ctx, cond));
    solver.assert(z3.Z3_mk_eq(pass.encoder.context.ctx, merged, one));
    solver.assert(z3.Z3_mk_not(pass.encoder.context.ctx, z3.Z3_mk_eq(pass.encoder.context.ctx, untouched_base, one)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "branch merge preserves conditional memref initialization" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    var base = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer base.deinit(testing.allocator);
    var then_state = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer then_state.deinit(testing.allocator);
    var else_state = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer else_state.deinit(testing.allocator);

    const bool_sort = z3.Z3_mk_bool_sort(pass.encoder.context.ctx);
    const i256_sort = z3.Z3_mk_bv_sort(pass.encoder.context.ctx, 256);
    const cond = try pass.encoder.mkVariable("memref_branch_cond", bool_sort);
    const forty_two = z3.Z3_mk_numeral(pass.encoder.context.ctx, "42", i256_sort);

    try then_state.memref_map.put(1234, .{
        .value = forty_two,
        .initialized = z3.Z3_mk_true(pass.encoder.context.ctx),
    });

    try pass.mergeEncoderBranchState(cond, &base, &then_state, &else_state);

    try testing.expect(!pass.encoder.isDegraded());
    const merged = pass.encoder.memref_map.get(1234).?;
    try testing.expect(z3.c.Z3_is_eq_ast(pass.encoder.context.ctx, merged.value, forty_two));
    try testing.expect(z3.c.Z3_is_eq_ast(pass.encoder.context.ctx, merged.initialized, cond));
}

test "many-branch merge preserves untouched global base state" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    var base = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer base.deinit(testing.allocator);
    var branch_states = [_]VerificationPass.EncoderBranchState{
        VerificationPass.EncoderBranchState.init(testing.allocator),
        VerificationPass.EncoderBranchState.init(testing.allocator),
        VerificationPass.EncoderBranchState.init(testing.allocator),
    };
    defer for (&branch_states) |*state| state.deinit(testing.allocator);

    const bool_sort = z3.Z3_mk_bool_sort(pass.encoder.context.ctx);
    const i256_sort = z3.Z3_mk_bv_sort(pass.encoder.context.ctx, 256);
    const conditions = [_]z3.Z3_ast{
        try pass.encoder.mkVariable("switch_c0", bool_sort),
        try pass.encoder.mkVariable("switch_c1", bool_sort),
        try pass.encoder.mkVariable("switch_c2", bool_sort),
    };
    const one = z3.Z3_mk_numeral(pass.encoder.context.ctx, "1", i256_sort);
    const two = z3.Z3_mk_numeral(pass.encoder.context.ctx, "2", i256_sort);

    try branch_states[0].global_map.put(try testing.allocator.dupe(u8, "counter"), one);
    try branch_states[2].global_map.put(try testing.allocator.dupe(u8, "counter"), two);

    try pass.mergeEncoderBranchStatesMany(&conditions, &base, &branch_states);

    const merged = pass.encoder.global_map.get("counter").?;
    const untouched_base = pass.encoder.global_entry_map.get("counter").?;

    var solver = try Solver.init(pass.encoder.context, testing.allocator);
    defer solver.deinit();
    for (conditions) |cond| {
        solver.assert(z3.Z3_mk_not(pass.encoder.context.ctx, cond));
    }
    solver.assert(z3.Z3_mk_eq(pass.encoder.context.ctx, merged, one));
    solver.assert(z3.Z3_mk_not(pass.encoder.context.ctx, z3.Z3_mk_eq(pass.encoder.context.ctx, untouched_base, one)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "many-branch merge degrades partial memref initialization" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    var base = VerificationPass.EncoderBranchState.init(testing.allocator);
    defer base.deinit(testing.allocator);
    var branch_states = [_]VerificationPass.EncoderBranchState{
        VerificationPass.EncoderBranchState.init(testing.allocator),
        VerificationPass.EncoderBranchState.init(testing.allocator),
        VerificationPass.EncoderBranchState.init(testing.allocator),
    };
    defer for (&branch_states) |*state| state.deinit(testing.allocator);

    const bool_sort = z3.Z3_mk_bool_sort(pass.encoder.context.ctx);
    const i256_sort = z3.Z3_mk_bv_sort(pass.encoder.context.ctx, 256);
    const conditions = [_]z3.Z3_ast{
        try pass.encoder.mkVariable("switch_mem_c0", bool_sort),
        try pass.encoder.mkVariable("switch_mem_c1", bool_sort),
        try pass.encoder.mkVariable("switch_mem_c2", bool_sort),
    };
    const seven = z3.Z3_mk_numeral(pass.encoder.context.ctx, "7", i256_sort);

    try branch_states[0].memref_map.put(777, .{
        .value = seven,
        .initialized = z3.Z3_mk_true(pass.encoder.context.ctx),
    });
    try branch_states[2].memref_map.put(777, .{
        .value = seven,
        .initialized = z3.Z3_mk_true(pass.encoder.context.ctx),
    });

    try pass.mergeEncoderBranchStatesMany(&conditions, &base, &branch_states);

    try testing.expect(!pass.encoder.isDegraded());
    try testing.expect(pass.encoder.memref_map.get(777) != null);
}

test "private callee obligations are guarded by callee requires in public summaries" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Basic);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPublicCallsPrivateRequiresGuardedAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var saw_public_obligation = false;
    for (queries.items) |q| {
        if (!std.mem.eql(u8, q.function_name, "public_entry")) continue;
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            saw_public_obligation = true;
            break;
        }
    }
    try testing.expect(saw_public_obligation);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "private result callee obligations are guarded by callee requires in public summaries" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Basic);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPublicCallsPrivateRequiresGuardedAssertReturningModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| q.deinit(testing.allocator);
        queries.deinit();
    }

    var saw_public_obligation = false;
    for (queries.items) |q| {
        if (!std.mem.eql(u8, q.function_name, "public_entry")) continue;
        if (q.kind == .Obligation and q.obligation_kind == .ContractInvariant) {
            saw_public_obligation = true;
            break;
        }
    }
    try testing.expect(saw_public_obligation);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "public callee obligations stay on the callee instead of bubbling to caller" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPublicCallsPublicAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var caller_obligations: usize = 0;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        if (std.mem.eql(u8, q.function_name, "public_entry")) caller_obligations += 1;
    }

    try testing.expectEqual(@as(usize, 0), caller_obligations);
}

test "path assume annotations are extracted as path assumptions" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPathAssumeEnsuresModule(mlir_ctx, 1, 1);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);

    var path_assume_count: usize = 0;
    var user_assume_count: usize = 0;
    for (pass.encoded_annotations.items) |ann| {
        switch (ann.kind) {
            .PathAssume => path_assume_count += 1,
            .Assume => user_assume_count += 1,
            else => {},
        }
    }
    try testing.expectEqual(@as(usize, 1), path_assume_count);
    try testing.expectEqual(@as(usize, 0), user_assume_count);
}

test "base query excludes path assumptions" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPathAssumeEnsuresModule(mlir_ctx, 0, 1);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_base = false;
    for (queries.items) |q| {
        if (q.kind != .Base) continue;
        found_base = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), status);
    }
    try testing.expect(found_base);
}

test "obligation query includes scoped path assumptions" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPathAssumeEnsuresModule(mlir_ctx, 0, 0);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .Ensures) continue;
        found_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_obligation);
}

test "prepared queries carry narrow tracked assumptions" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPathAssumeEnsuresModule(mlir_ctx, 0, 0);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var saw_base = false;
    var saw_obligation = false;

    for (queries.items) |q| {
        if (!std.mem.eql(u8, q.function_name, "path_assume_test")) continue;
        switch (q.kind) {
            .Base => {
                saw_base = true;
                try testing.expectEqual(@as(usize, 0), q.tracked_assumptions.len);
            },
            .Obligation => {
                if (q.obligation_kind != .Ensures) continue;
                saw_obligation = true;
                try testing.expectEqual(@as(usize, 2), q.tracked_assumptions.len);
                try testing.expectEqual(AssumptionKind.path_assume, q.tracked_assumptions[0].tag.kind);
                try testing.expectEqualStrings("path assumption", q.tracked_assumptions[0].tag.label);
                try testing.expectEqual(AssumptionKind.goal, q.tracked_assumptions[1].tag.kind);
                try testing.expectEqualStrings("goal", q.tracked_assumptions[1].tag.label);
            },
            else => {},
        }
    }

    try testing.expect(saw_base);
    try testing.expect(saw_obligation);
}

test "guard violate query includes scoped path assumptions" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildLinearPathAssumeGuardModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_violate = false;
    for (queries.items) |q| {
        if (q.kind != .GuardViolate) continue;
        found_violate = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_violate);
}

test "guard satisfy queries ignore incompatible sibling branch paths" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildBranchPathGuardsModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var satisfy_count: usize = 0;
    for (queries.items) |q| {
        if (q.kind != .GuardSatisfy) continue;
        satisfy_count += 1;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), status);
    }

    try testing.expectEqual(@as(usize, 2), satisfy_count);
}

test "sequential guard verification ignores sibling branch guards" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.parallel = false;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildBranchPathGuardsModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "sequential guard verification uses linear path assumptions to prove guards" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.parallel = false;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildLinearPathAssumeGuardModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var result = try pass.runVerificationPass(module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expect(result.proven_guard_ids.contains("guard:test:linear"));
}

test "path constraint normalization flattens boolean ite ladders" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const bool_sort = z3.Z3_mk_bool_sort(pass.context.ctx);
    const cond = try pass.encoder.mkVariable("ite_ladder_cond", bool_sort);
    const prev = try pass.encoder.mkVariable("ite_ladder_prev", bool_sort);

    const ite = z3.Z3_mk_ite(pass.context.ctx, cond, prev, z3.Z3_mk_true(pass.context.ctx));
    const nested = z3.Z3_mk_not(pass.context.ctx, ite);

    try pass.active_path_assumptions.append(.{
        .condition = nested,
        .extra_constraints = &[_]z3.Z3_ast{},
        .owned_extra_constraints = false,
    });

    const constraints = try pass.captureActivePathConstraints();
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    try testing.expect(constraints.len > 0);

    var solver = try Solver.init(pass.encoder.context, testing.allocator);
    defer solver.deinit();
    for (constraints) |constraint| solver.assert(constraint);
    solver.assert(z3.Z3_mk_not(pass.context.ctx, nested));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

fn verificationErrorTypeCounts(result: *const errors.VerificationResult) [@typeInfo(errors.VerificationErrorType).@"enum".fields.len]usize {
    var counts = [_]usize{0} ** @typeInfo(errors.VerificationErrorType).@"enum".fields.len;
    for (result.errors.items) |err| {
        counts[@intFromEnum(err.error_type)] += 1;
    }
    return counts;
}

fn expectVerificationResultsEquivalent(lhs: *const errors.VerificationResult, rhs: *const errors.VerificationResult) !void {
    try testing.expectEqual(lhs.success, rhs.success);
    try testing.expectEqual(lhs.errors.items.len, rhs.errors.items.len);
    try testing.expectEqual(lhs.diagnostics.items.len, rhs.diagnostics.items.len);
    try testing.expectEqualDeep(verificationErrorTypeCounts(lhs), verificationErrorTypeCounts(rhs));
}

test "parallel verification matches sequential verification on guard-success module" {
    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildBranchPathGuardsModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var sequential = try VerificationPass.init(testing.allocator);
    defer sequential.deinit();
    sequential.parallel = false;

    var parallel = try VerificationPass.init(testing.allocator);
    defer parallel.deinit();
    parallel.parallel = true;

    var seq_result = try sequential.runVerificationPass(module);
    defer seq_result.deinit();
    var par_result = try parallel.runVerificationPass(module);
    defer par_result.deinit();

    try expectVerificationResultsEquivalent(&seq_result, &par_result);
}

test "parallel verification matches sequential verification on failing module" {
    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildPublicCallsPrivateAssertModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var sequential = try VerificationPass.init(testing.allocator);
    defer sequential.deinit();
    sequential.parallel = false;
    sequential.setVerifyMode(.Basic);

    var parallel = try VerificationPass.init(testing.allocator);
    defer parallel.deinit();
    parallel.parallel = true;
    parallel.setVerifyMode(.Basic);

    var seq_result = try sequential.runVerificationPass(module);
    defer seq_result.deinit();
    var par_result = try parallel.runVerificationPass(module);
    defer par_result.deinit();

    try expectVerificationResultsEquivalent(&seq_result, &par_result);
}

test "parallel verification matches sequential verification on loop invariant module" {
    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildForContractInvariantModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var sequential = try VerificationPass.init(testing.allocator);
    defer sequential.deinit();
    sequential.parallel = false;
    sequential.setVerifyMode(.Full);

    var parallel = try VerificationPass.init(testing.allocator);
    defer parallel.deinit();
    parallel.parallel = true;
    parallel.setVerifyMode(.Full);

    var seq_result = try sequential.runVerificationPass(module);
    defer seq_result.deinit();
    var par_result = try parallel.runVerificationPass(module);
    defer par_result.deinit();

    try expectVerificationResultsEquivalent(&seq_result, &par_result);
}

test "parallel verification matches sequential verification on conditional return stateful module" {
    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildConditionalReturnStatefulDivModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    var sequential = try VerificationPass.init(testing.allocator);
    defer sequential.deinit();
    sequential.parallel = false;
    sequential.setVerifyMode(.Full);

    var parallel = try VerificationPass.init(testing.allocator);
    defer parallel.deinit();
    parallel.parallel = true;
    parallel.setVerifyMode(.Full);

    var seq_result = try sequential.runVerificationPass(module);
    defer seq_result.deinit();
    var par_result = try parallel.runVerificationPass(module);
    defer par_result.deinit();

    try expectVerificationResultsEquivalent(&seq_result, &par_result);
}

test "scaledParallelTimeoutMs widens timeout by worker count" {
    try testing.expectEqual(@as(?u32, null), scaledParallelTimeoutMs(null, 4));
    try testing.expectEqual(@as(?u32, 1000), scaledParallelTimeoutMs(1000, 1));
    try testing.expectEqual(@as(?u32, 4000), scaledParallelTimeoutMs(1000, 4));
    try testing.expectEqual(@as(?u32, std.math.maxInt(u32)), scaledParallelTimeoutMs(std.math.maxInt(u32), 2));
}

test "contract invariants from loop body obligations use loop constraints" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildForContractInvariantModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "contract invariants inside ora.conditional_return use path constraints" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildConditionalReturnContractInvariantModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "contract invariants after ora.conditional_return use fallthrough path constraints" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildConditionalReturnFallthroughModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "contract invariants after sequential ora.conditional_return accumulate fallthrough path constraints" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildDoubleConditionalReturnFallthroughModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "stateful obligations after ora.conditional_return use fallthrough path constraints" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildConditionalReturnStatefulDivModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "tstore obligations after ora.conditional_return use fallthrough path constraints" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildConditionalReturnTStoreDivModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "map_store obligations after ora.conditional_return use fallthrough path constraints" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildConditionalReturnMapStoreDivModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "map_get-div-map_store obligations respect conditional_return fallthrough" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setVerifyMode(.Full);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    testLoadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const module = buildConditionalReturnMapGetDivModule(mlir_ctx);
    defer mlir.oraModuleDestroy(module);

    try pass.extractAnnotationsFromMLIR(module);
    var queries = try pass.buildPreparedQueries();
    defer {
        for (queries.items) |*q| {
            q.deinit(testing.allocator);
        }
        queries.deinit();
    }

    var found_contract_obligation = false;
    for (queries.items) |q| {
        if (q.kind != .Obligation or q.obligation_kind != .ContractInvariant) continue;
        found_contract_obligation = true;
        pass.solver.reset();
        try assertPreparedQueryConstraints(&pass.solver, q.constraints);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_contract_obligation);
}

test "degraded SMT encoding fails closed" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    pass.encoder.encoding_degraded = true;
    pass.encoder.encoding_degraded_reason = "test degradation";

    var result = try pass.degradedVerificationResult();
    defer result.deinit();

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors.items.len);
    try testing.expectEqual(errors.VerificationErrorType.EncodingDegraded, result.errors.items[0].error_type);
    try testing.expect(std.mem.containsAtLeast(u8, result.errors.items[0].message, 1, "SMT encoding degraded"));
    try testing.expect(std.mem.containsAtLeast(u8, result.errors.items[0].message, 1, "test degradation"));
}

test "mid-proof SMT degradation fails closed" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    var result = errors.VerificationResult.init(testing.allocator);
    defer result.deinit();

    pass.encoder.encoding_degraded = true;
    pass.encoder.encoding_degraded_reason = "mid-proof degradation";

    try pass.addDegradedDuringProvingError(&result, "/tmp/test.ora", 9, 4);

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors.items.len);
    try testing.expectEqual(errors.VerificationErrorType.EncodingDegraded, result.errors.items[0].error_type);
    try testing.expectEqualStrings("/tmp/test.ora", result.errors.items[0].file);
    try testing.expectEqual(@as(u32, 9), result.errors.items[0].line);
    try testing.expectEqual(@as(u32, 4), result.errors.items[0].column);
    try testing.expect(std.mem.containsAtLeast(u8, result.errors.items[0].message, 1, "degraded during proving"));
    try testing.expect(std.mem.containsAtLeast(u8, result.errors.items[0].message, 1, "mid-proof degradation"));
}

test "unknown verification errors fail closed" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    var result = errors.VerificationResult.init(testing.allocator);
    defer result.deinit();

    try pass.addUnknownVerificationError(
        &result,
        try testing.allocator.dupe(u8, "verification assumptions are unknown in f"),
        "/tmp/test.ora",
        7,
        3,
    );

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 1), result.errors.items.len);
    try testing.expectEqual(errors.VerificationErrorType.Unknown, result.errors.items[0].error_type);
    try testing.expectEqualStrings("/tmp/test.ora", result.errors.items[0].file);
    try testing.expectEqual(@as(u32, 7), result.errors.items[0].line);
    try testing.expectEqual(@as(u32, 3), result.errors.items[0].column);
}

test "parallel result collection taints function on unknown base" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const queries = [_]PreparedQuery{
        .{
            .kind = .Base,
            .function_name = "f",
            .file = "/tmp/test.ora",
            .line = 1,
            .column = 1,
            .smtlib_z = try testing.allocator.dupeZ(u8, "(check-sat)"),
            .log_prefix = try testing.allocator.dupe(u8, "verification: f [base]"),
        },
        .{
            .kind = .Obligation,
            .function_name = "f",
            .obligation_kind = .ContractInvariant,
            .file = "/tmp/test.ora",
            .line = 2,
            .column = 1,
            .smtlib_z = try testing.allocator.dupeZ(u8, "(check-sat)"),
            .log_prefix = try testing.allocator.dupe(u8, "verification: f [contract invariant]"),
        },
    };
    defer {
        var mutable_queries = queries;
        for (&mutable_queries) |*query| query.deinit(testing.allocator);
    }

    const results = [_]PreparedQueryResult{
        .{ .status = z3.Z3_L_UNDEF },
        .{ .status = z3.Z3_L_TRUE },
    };

    var collected = try pass.collectPreparedQueryResults(queries[0..], results[0..]);
    defer collected.deinit();

    try testing.expect(!collected.success);
    try testing.expectEqual(@as(usize, 1), collected.errors.items.len);
    try testing.expectEqual(errors.VerificationErrorType.Unknown, collected.errors.items[0].error_type);
    try testing.expect(std.mem.containsAtLeast(u8, collected.errors.items[0].message, 1, "verification assumptions are unknown in f"));
}

test "parallel result collection skips obligations on inconsistent base" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const queries = [_]PreparedQuery{
        .{
            .kind = .Base,
            .function_name = "f",
            .file = "/tmp/test.ora",
            .line = 1,
            .column = 1,
            .smtlib_z = try testing.allocator.dupeZ(u8, "(check-sat)"),
            .log_prefix = try testing.allocator.dupe(u8, "verification: f [base]"),
        },
        .{
            .kind = .Obligation,
            .function_name = "f",
            .obligation_kind = .ContractInvariant,
            .file = "/tmp/test.ora",
            .line = 2,
            .column = 1,
            .smtlib_z = try testing.allocator.dupeZ(u8, "(check-sat)"),
            .log_prefix = try testing.allocator.dupe(u8, "verification: f [contract invariant]"),
        },
    };
    defer {
        var mutable_queries = queries;
        for (&mutable_queries) |*query| query.deinit(testing.allocator);
    }

    const results = [_]PreparedQueryResult{
        .{ .status = z3.Z3_L_FALSE },
        .{ .status = z3.Z3_L_TRUE },
    };

    var collected = try pass.collectPreparedQueryResults(queries[0..], results[0..]);
    defer collected.deinit();

    try testing.expect(!collected.success);
    try testing.expectEqual(@as(usize, 1), collected.errors.items.len);
    try testing.expectEqual(errors.VerificationErrorType.PreconditionViolation, collected.errors.items[0].error_type);
    try testing.expect(std.mem.containsAtLeast(u8, collected.errors.items[0].message, 1, "verification assumptions are inconsistent in f"));
}

test "report success is false without verification result when queries are unknown" {
    const summary = ReportSummary{
        .total_queries = 1,
        .unknown = 1,
    };
    try testing.expect(!inferReportVerificationSuccess(summary, null, false));
}

test "report success follows verification result when present" {
    var result = errors.VerificationResult.init(testing.allocator);
    defer result.deinit();
    result.success = false;

    const summary = ReportSummary{
        .total_queries = 1,
        .unknown = 0,
    };
    try testing.expect(!inferReportVerificationSuccess(summary, &result, false));
}

test "report success is false when encoder degraded" {
    const summary = ReportSummary{
        .total_queries = 1,
        .unknown = 0,
    };
    try testing.expect(!inferReportVerificationSuccess(summary, null, true));
}

test "rendered SMT report json includes degradation metadata" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const summary = ReportSummary{
        .encoding_degraded = true,
        .degradation_reason = "test degradation",
    };
    const kind_counts = ReportKindCounts{};

    const json = try pass.renderSmtReportJson(
        "/tmp/test.ora",
        0,
        &.{},
        &.{},
        summary,
        kind_counts,
        null,
    );
    defer testing.allocator.free(json);

    try testing.expect(std.mem.indexOf(u8, json, "\"encoding_degraded\":true") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"degradation_reason\":\"test degradation\"") != null);
}

test "rendered SMT report json includes vacuous explain tags" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    const query = PreparedQuery{
        .kind = .Obligation,
        .function_name = "f",
        .obligation_kind = .Ensures,
        .file = "/tmp/test.ora",
        .line = 12,
        .column = 3,
        .smtlib_z = try testing.allocator.dupeZ(u8, "(check-sat)"),
        .log_prefix = try testing.allocator.dupe(u8, "f [ensures]"),
    };
    defer {
        var mutable_query = query;
        mutable_query.deinit(testing.allocator);
    }

    const explain_tags = [_]AssumptionTag{
        .{
            .kind = .requires,
            .function_name = "f",
            .file = "/tmp/test.ora",
            .line = 10,
            .column = 3,
            .label = "requires",
        },
        .{
            .kind = .goal,
            .function_name = "f",
            .file = "/tmp/test.ora",
            .line = 12,
            .column = 3,
            .label = "goal",
        },
    };
    const run = ReportQueryRun{
        .status = z3.Z3_L_FALSE,
        .elapsed_ms = 1,
        .explain = "requires test.ora:10 requires; goal test.ora:12 goal",
        .explain_tags = explain_tags[0..],
        .vacuous = true,
    };
    const summary = ReportSummary{
        .total_queries = 1,
        .unsat = 1,
        .verification_success = true,
    };
    const kind_counts = ReportKindCounts{
        .obligation = 1,
    };

    const json = try pass.renderSmtReportJson(
        "/tmp/test.ora",
        0,
        (&[_]PreparedQuery{query})[0..],
        (&[_]ReportQueryRun{run})[0..],
        summary,
        kind_counts,
        null,
    );
    defer testing.allocator.free(json);

    try testing.expect(std.mem.indexOf(u8, json, "\"vacuous\":true") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"explain_core\":\"requires test.ora:10 requires; goal test.ora:12 goal\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"explain_tags\":[{\"kind\":\"requires\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"kind\":\"goal\"") != null);
}

test "rendered SMT report json includes multi-requires explain tags and summary vacuous count" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();
    pass.setExplainCores(true);

    const query = PreparedQuery{
        .kind = .Obligation,
        .function_name = "g",
        .obligation_kind = .ContractInvariant,
        .file = "/tmp/test.ora",
        .line = 20,
        .column = 5,
        .smtlib_z = try testing.allocator.dupeZ(u8, "(check-sat)"),
        .log_prefix = try testing.allocator.dupe(u8, "g [contract invariant]"),
    };
    defer {
        var mutable_query = query;
        mutable_query.deinit(testing.allocator);
    }

    const explain_tags = [_]AssumptionTag{
        .{ .kind = .requires, .function_name = "g", .file = "/tmp/test.ora", .line = 14, .column = 5, .label = "requires" },
        .{ .kind = .requires, .function_name = "g", .file = "/tmp/test.ora", .line = 15, .column = 5, .label = "requires" },
        .{ .kind = .goal, .function_name = "g", .file = "/tmp/test.ora", .line = 20, .column = 5, .label = "checked addition overflow" },
    };
    const run = ReportQueryRun{
        .status = z3.Z3_L_FALSE,
        .elapsed_ms = 2,
        .explain = "requires test.ora:14 requires; requires test.ora:15 requires; goal test.ora:20 checked addition overflow",
        .explain_tags = explain_tags[0..],
        .vacuous = false,
    };
    const summary = ReportSummary{
        .total_queries = 1,
        .unsat = 1,
        .vacuous = 1,
        .verification_success = true,
    };
    const kind_counts = ReportKindCounts{
        .obligation = 1,
    };

    const json = try pass.renderSmtReportJson(
        "/tmp/test.ora",
        0,
        (&[_]PreparedQuery{query})[0..],
        (&[_]ReportQueryRun{run})[0..],
        summary,
        kind_counts,
        null,
    );
    defer testing.allocator.free(json);

    try testing.expect(std.mem.indexOf(u8, json, "\"explain_cores\":true") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"vacuous\":1") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"line\":14") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"line\":15") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"label\":\"checked addition overflow\"") != null);
}

test "buildDegradedSmtReport emits degraded report with no queries" {
    var pass = try VerificationPass.init(testing.allocator);
    defer pass.deinit();

    pass.encoder.encoding_degraded = true;
    pass.encoder.encoding_degraded_reason = "test degradation";

    var report = try pass.buildDegradedSmtReport("/tmp/test.ora", null);
    defer report.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, report.markdown, "Encoding degraded: `true`") != null);
    try testing.expect(std.mem.indexOf(u8, report.markdown, "test degradation") != null);
    try testing.expect(std.mem.indexOf(u8, report.json, "\"encoding_degraded\":true") != null);
    try testing.expect(std.mem.indexOf(u8, report.json, "\"total_queries\":0") != null);
}

test "parseModelString preserves user names with double underscore prefix" {
    const model =
        "__admin -> #x01\n" ++
        "__ora_internal -> #x02\n" ++
        "undef_tmp -> #x03\n";

    const ce = parseModelString(testing.allocator, model) orelse return error.TestUnexpectedResult;
    defer {
        var mutable = ce;
        mutable.deinit();
    }

    try testing.expectEqual(@as(usize, 1), ce.variables.count());
    try testing.expectEqualStrings("#x01", ce.variables.get("__admin").?);
    try testing.expect(ce.variables.get("__ora_internal") == null);
    try testing.expect(ce.variables.get("undef_tmp") == null);
}

test "unknown errors map back to unknown queries in reports" {
    const query = PreparedQuery{
        .kind = .Obligation,
        .function_name = "f",
        .obligation_kind = .ContractInvariant,
        .file = "/tmp/test.ora",
        .line = 9,
        .column = 2,
        .smtlib_z = try testing.allocator.dupeZ(u8, "(check-sat)"),
        .log_prefix = try testing.allocator.dupe(u8, "verification: f [contract invariant]"),
    };
    defer {
        var mutable_query = query;
        mutable_query.deinit(testing.allocator);
    }

    const run = ReportQueryRun{
        .status = z3.Z3_L_UNDEF,
        .elapsed_ms = 1,
    };

    var err_result = errors.VerificationResult.init(testing.allocator);
    defer err_result.deinit();
    try err_result.addError(.{
        .error_type = .Unknown,
        .message = try testing.allocator.dupe(u8, "could not prove contract invariant in f"),
        .file = try testing.allocator.dupe(u8, "/tmp/test.ora"),
        .line = 9,
        .column = 2,
        .counterexample = null,
        .allocator = testing.allocator,
    });

    const idx = findQueryIndexForError(err_result.errors.items[0], (&[_]PreparedQuery{query})[0..], (&[_]ReportQueryRun{run})[0..]);
    try testing.expectEqual(@as(?usize, 0), idx);

    const classification = classifyQueryFailure(query, run);
    try testing.expectEqualStrings("UnknownObligation", classification.subtype.?);
}

test "parseLocationString strips mlir loc wrapper" {
    const parsed = parseLocationString("loc(\"/tmp/example.ora\":42:7)");
    try testing.expectEqualStrings("/tmp/example.ora", parsed.file);
    try testing.expectEqual(@as(u32, 42), parsed.line);
    try testing.expectEqual(@as(u32, 7), parsed.column);
}

test "parseLocationString prefers innermost file location from nested ora tags" {
    const parsed = parseLocationString("loc(\"ora.stmt.0\"(\"ora.origin_stmt.0\"(\"test.ora\":3:5)))");
    try testing.expectEqualStrings("test.ora", parsed.file);
    try testing.expectEqual(@as(u32, 3), parsed.line);
    try testing.expectEqual(@as(u32, 5), parsed.column);
}
