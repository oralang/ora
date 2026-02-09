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
const lib = @import("ora_lib");
const Context = @import("context.zig").Context;
const Solver = @import("solver.zig").Solver;
const Encoder = @import("encoder.zig").Encoder;
const errors = @import("errors.zig");
const ManagedArrayList = std.array_list.Managed;

/// Verification annotation extracted from AST
pub const VerificationAnnotation = struct {
    kind: AnnotationKind,
    condition: *lib.ast.Expressions.ExprNode,
    source_location: lib.ast.SourceSpan,

    pub const AnnotationKind = enum {
        Requires, // Function precondition
        Ensures, // Function postcondition
        LoopInvariant, // Loop invariant
        ContractInvariant, // Contract-level invariant (future)
        RefinementGuard, // Runtime refinement guard
        Assume, // Verification-only assumption
        PathAssume, // Compiler-injected path-local assumption
    };
};

/// Collection of verification annotations for a function
pub const FunctionAnnotations = struct {
    function_name: []const u8,
    requires: []*lib.ast.Expressions.ExprNode,
    ensures: []*lib.ast.Expressions.ExprNode,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, function_name: []const u8) FunctionAnnotations {
        return .{
            .function_name = function_name,
            .requires = &[_]*lib.ast.Expressions.ExprNode{},
            .ensures = &[_]*lib.ast.Expressions.ExprNode{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FunctionAnnotations) void {
        // note: We don't own the ExprNode pointers, they're from AST arena
        _ = self;
    }
};

/// Collection of loop invariants
pub const LoopInvariants = struct {
    invariants: []*lib.ast.Expressions.ExprNode,
    source_location: lib.ast.SourceSpan,

    pub fn init(invariants: []*lib.ast.Expressions.ExprNode, span: lib.ast.SourceSpan) LoopInvariants {
        return .{
            .invariants = invariants,
            .source_location = span,
        };
    }
};

/// Verification pass for MLIR modules
pub const VerificationPass = struct {
    pub const VerifyMode = enum {
        Basic,
        Full,
    };

    context: *Context,
    solver: Solver,
    encoder: Encoder,
    allocator: std.mem.Allocator,
    debug_z3: bool,
    timeout_ms: ?u32,
    parallel: bool,
    max_workers: usize,
    filter_function_name: ?[]const u8 = null,
    verify_mode: VerifyMode = .Basic,
    verify_calls: bool = true,
    verify_state: bool = true,
    verify_stats: bool = false,

    /// Map from function name to its annotations
    function_annotations: std.StringHashMap(FunctionAnnotations),

    /// List of loop invariants (with their locations)
    loop_invariants: ManagedArrayList(LoopInvariants),

    /// Current function being processed (for MLIR extraction)
    current_function_name: ?[]const u8 = null,

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

        const timeout_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_TIMEOUT_MS") catch null;
        var timeout_ms: ?u32 = 60_000;
        if (timeout_env) |val| {
            timeout_ms = std.fmt.parseInt(u32, val, 10) catch null;
            allocator.free(val);
        }

        const parallel_env = std.process.getEnvVarOwned(allocator, "ORA_Z3_PARALLEL") catch null;
        const parallel = if (parallel_env) |val| blk: {
            defer allocator.free(val);
            break :blk parseBoolEnv(val);
        } else true;

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
        } else .Basic;

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

        encoder.setVerifyCalls(verify_calls);
        encoder.setVerifyState(verify_state);

        if (timeout_ms) |ms| {
            solver.setTimeoutMs(ms);
        }

        const pass = VerificationPass{
            .context = context,
            .solver = solver,
            .encoder = encoder,
            .allocator = allocator,
            .debug_z3 = debug_z3,
            .timeout_ms = timeout_ms,
            .parallel = parallel,
            .max_workers = max_workers,
            .verify_mode = verify_mode,
            .verify_calls = verify_calls,
            .verify_state = verify_state,
            .verify_stats = verify_stats,
            .function_annotations = std.StringHashMap(FunctionAnnotations).init(allocator),
            .loop_invariants = ManagedArrayList(LoopInvariants).init(allocator),
            .encoded_annotations = ManagedArrayList(EncodedAnnotation).init(allocator),
            .active_path_assumptions = ManagedArrayList(ActivePathAssume).init(allocator),
            .function_name_storage = ManagedArrayList([]const u8).init(allocator),
            .location_storage = ManagedArrayList([]const u8).init(allocator),
            .guard_id_storage = ManagedArrayList([]const u8).init(allocator),
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

    pub fn deinit(self: *VerificationPass) void {
        var iter = self.function_annotations.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.function_annotations.deinit();
        self.loop_invariants.deinit();
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
        for (self.encoded_annotations.items) |ann| {
            if (ann.extra_constraints.len > 0) {
                self.allocator.free(ann.extra_constraints);
            }
            if (ann.old_extra_constraints.len > 0) {
                self.allocator.free(ann.old_extra_constraints);
            }
            if (ann.loop_step_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_step_extra_constraints);
            }
            if (ann.loop_exit_extra_constraints.len > 0) {
                self.allocator.free(ann.loop_exit_extra_constraints);
            }
            if (ann.path_constraints.len > 0) {
                self.allocator.free(ann.path_constraints);
            }
        }
        self.encoded_annotations.deinit();
        self.active_path_assumptions.deinit();
        self.encoder.deinit();
        self.solver.deinit();
        self.context.deinit();
        self.allocator.destroy(self.context);
    }

    //===----------------------------------------------------------------------===//
    // annotation Extraction from AST
    //===----------------------------------------------------------------------===//

    /// Extract verification annotations from AST module
    pub fn extractAnnotationsFromAST(self: *VerificationPass, module: *lib.ast.AstNode) !void {
        switch (module.*) {
            .Module => |*mod| {
                // walk all top-level declarations
                for (mod.declarations) |*decl| {
                    try self.extractAnnotationsFromDeclaration(decl);
                }
            },
            else => {
                // not a module, try to extract from the node directly
                try self.extractAnnotationsFromDeclaration(module);
            },
        }
    }

    /// Extract annotations from a declaration (contract, function, etc.)
    fn extractAnnotationsFromDeclaration(self: *VerificationPass, decl: *lib.ast.AstNode) !void {
        switch (decl.*) {
            .Contract => |*contract| {
                // extract annotations from contract members
                for (contract.body) |*member| {
                    try self.extractAnnotationsFromDeclaration(member);
                }
            },
            .Function => |*function| {
                // include ghost functions in verification (they're specification-only)
                // ghost functions are used for verification but not compiled to bytecode
                try self.extractFunctionAnnotations(function);
            },
            else => {
                // other declaration types don't have verification annotations
            },
        }
    }

    /// Extract requires/ensures clauses from a function node
    fn extractFunctionAnnotations(self: *VerificationPass, function: *lib.ast.FunctionNode) !void {
        const function_name = function.name;

        // create function annotations entry
        var annotations = FunctionAnnotations.init(self.allocator, function_name);

        // extract requires clauses
        if (function.requires_clauses.len > 0) {
            annotations.requires = try self.allocator.dupe(*lib.ast.Expressions.ExprNode, function.requires_clauses);
        }

        // extract ensures clauses
        if (function.ensures_clauses.len > 0) {
            annotations.ensures = try self.allocator.dupe(*lib.ast.Expressions.ExprNode, function.ensures_clauses);
        }

        // store in map
        const name_copy = try self.allocator.dupe(u8, function_name);
        try self.function_annotations.put(name_copy, annotations);
    }

    /// Extract loop invariants from a while statement
    pub fn extractLoopInvariantsFromWhile(self: *VerificationPass, while_stmt: *lib.ast.Statements.WhileNode) !void {
        if (while_stmt.invariants.len > 0) {
            const invariants_copy = try self.allocator.dupe(*lib.ast.Expressions.ExprNode, while_stmt.invariants);
            const loop_invariants = LoopInvariants.init(invariants_copy, while_stmt.span);
            try self.loop_invariants.append(loop_invariants);
        }
    }

    /// Extract loop invariants from a for loop statement
    pub fn extractLoopInvariantsFromFor(self: *VerificationPass, for_stmt: *lib.ast.Statements.ForLoopNode) !void {
        if (for_stmt.invariants.len > 0) {
            const invariants_copy = try self.allocator.dupe(*lib.ast.Expressions.ExprNode, for_stmt.invariants);
            const loop_invariants = LoopInvariants.init(invariants_copy, for_stmt.span);
            try self.loop_invariants.append(loop_invariants);
        }
    }

    /// Get annotations for a specific function
    pub fn getFunctionAnnotations(self: *VerificationPass, function_name: []const u8) ?*FunctionAnnotations {
        if (self.function_annotations.getPtr(function_name)) |annotations| {
            return annotations;
        }
        return null;
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
        // get first block in region
        var current_block = mlir.oraRegionGetFirstBlock(region);

        while (!mlir.oraBlockIsNull(current_block)) {
            const block_path_assumption_len = self.active_path_assumptions.items.len;
            defer self.active_path_assumptions.shrinkRetainingCapacity(block_path_assumption_len);

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
                if (std.mem.eql(u8, op_name, "func.func")) {
                    if (try self.getFunctionNameFromOp(current_op)) |fn_name| {
                        self.current_function_name = fn_name;
                        self.encoder.resetFunctionState();
                    }
                }

                try self.processMLIROperation(current_op);

                // walk nested regions (for functions, if statements, etc.)
                const num_regions = mlir.oraOperationGetNumRegions(current_op);
                for (0..@intCast(num_regions)) |region_idx| {
                    const nested_region = mlir.oraOperationGetRegion(current_op, @intCast(region_idx));
                    try self.walkMLIRRegion(nested_region);
                }

                if (std.mem.eql(u8, op_name, "func.func")) {
                    self.current_function_name = prev_function;
                }

                current_op = mlir.oraOperationGetNextInBlock(current_op);
            }

            current_block = mlir.oraBlockGetNextInRegion(current_block);
        }
    }

    /// Process a single MLIR operation to extract verification annotations
    fn processMLIROperation(self: *VerificationPass, op: mlir.MlirOperation) !void {
        const op_name_ref = self.getMLIROperationName(op);
        defer @import("mlir_c_api").freeStringRef(op_name_ref);
        const op_name = if (op_name_ref.data == null or op_name_ref.length == 0)
            ""
        else
            op_name_ref.data[0..op_name_ref.length];

        if (self.filter_function_name) |target_fn| {
            const current = self.current_function_name orelse return;
            if (!std.mem.eql(u8, current, target_fn)) return;
        }

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
                _ = try self.recordEncodedAnnotation(op, .LoopInvariant, condition_value, null);
            }
        } else if (std.mem.eql(u8, op_name, "ora.refinement_guard")) {
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                const guard_id = try self.getStringAttr(op, "ora.guard_id");
                _ = try self.recordEncodedAnnotation(op, .RefinementGuard, condition_value, guard_id);
            }
        } else if (std.mem.eql(u8, op_name, "ora.assume")) {
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                const origin_attr = try self.getStringAttr(op, "ora.assume_origin");
                const assume_kind: VerificationAnnotation.AnnotationKind = if (origin_attr) |origin| blk: {
                    if (std.mem.eql(u8, origin, "path")) break :blk .PathAssume;
                    break :blk .Assume;
                } else blk: {
                    const context_attr = try self.getStringAttr(op, "ora.verification_context");
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
        } else if (std.mem.eql(u8, op_name, "cf.assert")) {
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
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
                    // In full mode, treat untagged cf.assert as proof obligations.
                    _ = try self.recordEncodedAnnotation(op, .ContractInvariant, condition_value, null);
                }
            }
        }
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

    fn getStringAttr(self: *VerificationPass, op: mlir.MlirOperation, name: []const u8) !?[]const u8 {
        const attr = mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate(name.ptr, name.len));
        if (mlir.oraAttributeIsNull(attr)) return null;
        const value_ref = mlir.oraStringAttrGetValue(attr);
        if (value_ref.data == null or value_ref.length == 0) return null;
        const value_slice = value_ref.data[0..value_ref.length];
        const dup = try self.allocator.dupe(u8, value_slice);
        try self.guard_id_storage.append(dup);
        return dup;
    }

    fn recordEncodedAnnotation(
        self: *VerificationPass,
        op: mlir.MlirOperation,
        kind: VerificationAnnotation.AnnotationKind,
        condition_value: mlir.MlirValue,
        guard_id: ?[]const u8,
    ) !usize {
        const function_name = self.current_function_name orelse "unknown";
        const encoded = try self.encoder.encodeValue(condition_value);
        const extra_constraints = try self.encoder.takeConstraints(self.allocator);
        const safety_obligations = try self.encoder.takeObligations(self.allocator);
        defer if (safety_obligations.len > 0) self.allocator.free(safety_obligations);
        const path_constraints = try self.captureActivePathConstraints();

        var old_condition: ?z3.Z3_ast = null;
        var old_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
        var loop_step_condition: ?z3.Z3_ast = null;
        var loop_step_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
        var loop_exit_condition: ?z3.Z3_ast = null;
        var loop_exit_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{};
        if (kind == .LoopInvariant) {
            old_condition = try self.encoder.encodeValueOld(condition_value);
            old_extra_constraints = try self.encoder.takeConstraints(self.allocator);
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
                    .loop_step_condition = null,
                    .loop_step_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_exit_condition = null,
                    .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                    .file = loc_old.file,
                    .line = loc_old.line,
                    .column = loc_old.column,
                    .guard_id = null,
                });
            }

            loop_step_condition = try self.encodeLoopContinueCondition(op);
            loop_step_extra_constraints = try self.encoder.takeConstraints(self.allocator);
            const loop_step_safety = try self.encoder.takeObligations(self.allocator);
            defer if (loop_step_safety.len > 0) self.allocator.free(loop_step_safety);
            for (loop_step_safety) |obligation| {
                const loc_step = try self.getLocationInfo(op);
                try self.encoded_annotations.append(.{
                    .function_name = function_name,
                    .kind = .ContractInvariant,
                    .condition = obligation,
                    .extra_constraints = &[_]z3.Z3_ast{},
                    .path_constraints = try self.cloneConstraintSlice(path_constraints),
                    .old_condition = null,
                    .old_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_step_condition = null,
                    .loop_step_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_exit_condition = null,
                    .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                    .file = loc_step.file,
                    .line = loc_step.line,
                    .column = loc_step.column,
                    .guard_id = null,
                });
            }

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
                    .loop_step_condition = null,
                    .loop_step_extra_constraints = &[_]z3.Z3_ast{},
                    .loop_exit_condition = null,
                    .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                    .file = loc_loop.file,
                    .line = loc_loop.line,
                    .column = loc_loop.column,
                    .guard_id = null,
                });
            }
        }

        const loc = try self.getLocationInfo(op);
        const annotation_index = self.encoded_annotations.items.len;
        try self.encoded_annotations.append(.{
            .function_name = function_name,
            .kind = kind,
            .condition = encoded,
            .extra_constraints = extra_constraints,
            .path_constraints = path_constraints,
            .old_condition = old_condition,
            .old_extra_constraints = old_extra_constraints,
            .loop_step_condition = loop_step_condition,
            .loop_step_extra_constraints = loop_step_extra_constraints,
            .loop_exit_condition = loop_exit_condition,
            .loop_exit_extra_constraints = loop_exit_extra_constraints,
            .file = loc.file,
            .line = loc.line,
            .column = loc.column,
            .guard_id = guard_id,
        });

        // Safety obligations emitted by the encoder (e.g. non-zero divisor,
        // multiplication overflow checks) are tracked as contract invariants.
        for (safety_obligations) |obligation| {
            try self.encoded_annotations.append(.{
                .function_name = function_name,
                .kind = .ContractInvariant,
                .condition = obligation,
                .extra_constraints = &[_]z3.Z3_ast{},
                .path_constraints = try self.cloneConstraintSlice(path_constraints),
                .old_condition = null,
                .old_extra_constraints = &[_]z3.Z3_ast{},
                .loop_step_condition = null,
                .loop_step_extra_constraints = &[_]z3.Z3_ast{},
                .loop_exit_condition = null,
                .loop_exit_extra_constraints = &[_]z3.Z3_ast{},
                .file = loc.file,
                .line = loc.line,
                .column = loc.column,
                .guard_id = null,
            });
        }
        return annotation_index;
    }

    fn captureActivePathConstraints(self: *VerificationPass) ![]const z3.Z3_ast {
        if (self.active_path_assumptions.items.len == 0) {
            return &[_]z3.Z3_ast{};
        }
        var constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
        defer constraints.deinit();

        for (self.active_path_assumptions.items) |assume| {
            try addConstraintSlice(&constraints, assume.extra_constraints);
            try constraints.append(assume.condition);
        }

        return try constraints.toOwnedSlice();
    }

    fn cloneConstraintSlice(self: *VerificationPass, constraints: []const z3.Z3_ast) ![]const z3.Z3_ast {
        if (constraints.len == 0) {
            return &[_]z3.Z3_ast{};
        }
        return try self.allocator.dupe(z3.Z3_ast, constraints);
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

            const unsigned_cmp = self.getScfForUnsignedCmp(loop_op);

            return self.buildNumericLt(iv_ast, ub_ast, unsigned_cmp);
        }

        return null;
    }

    fn getScfForUnsignedCmp(_: *VerificationPass, loop_op: mlir.MlirOperation) bool {
        const printed = mlir.oraOperationPrintToString(loop_op);
        defer if (printed.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(printed);
        };

        if (printed.data != null and printed.length > 0) {
            const text = printed.data[0..printed.length];
            if (std.mem.indexOf(u8, text, "unsignedCmp = true") != null) return true;
            if (std.mem.indexOf(u8, text, "unsignedCmp = false") != null) return false;
        }

        const unsigned_attr = mlir.oraOperationGetAttributeByName(
            loop_op,
            mlir.oraStringRefCreate("unsignedCmp".ptr, "unsignedCmp".len),
        );
        if (mlir.oraAttributeIsNull(unsigned_attr)) return false;
        return mlir.oraIntegerAttrGetValueSInt(unsigned_attr) != 0;
    }

    fn encodeLoopExitCondition(self: *VerificationPass, invariant_op: mlir.MlirOperation) !?z3.Z3_ast {
        const continue_condition = try self.encodeLoopContinueCondition(invariant_op) orelse return null;
        return z3.Z3_mk_not(self.context.ctx, continue_condition);
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

    fn logAst(self: *VerificationPass, label: []const u8, ast: z3.Z3_ast) void {
        if (!self.debug_z3) return;
        const raw = z3.Z3_ast_to_string(self.context.ctx, ast);
        if (raw == null) return;
        const c_str: [*:0]const u8 = @ptrCast(raw);
        std.debug.print("[Z3] {s}: {s}\n", .{ label, std.mem.span(c_str) });
    }

    pub fn runVerificationPass(self: *VerificationPass, mlir_module: mlir.MlirModule) !errors.VerificationResult {
        if (self.parallel) {
            return try self.runVerificationPassParallel(mlir_module);
        }

        try self.extractAnnotationsFromMLIR(mlir_module);
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

        for (self.encoded_annotations.items) |ann| {
            const entry = try by_function.getOrPut(ann.function_name);
            if (!entry.found_existing) {
                entry.value_ptr.* = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            }
            try entry.value_ptr.append(ann);
        }

        var it = by_function.iterator();
        while (it.next()) |entry| {
            const fn_name = entry.key_ptr.*;
            const annotations = entry.value_ptr.items;
            if (annotations.len == 0) continue;
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
                    guard_annotations.append(ann) catch {};
                } else if (ann.kind == .PathAssume) {
                    path_assumption_annotations.append(ann) catch {};
                } else if (isObligationKind(ann.kind)) {
                    obligation_annotations.append(ann) catch {};
                    if (ann.kind == .Ensures) {
                        ensure_annotations.append(ann) catch {};
                    } else if (ann.kind == .LoopInvariant and ann.loop_exit_condition != null) {
                        loop_post_invariant_annotations.append(ann) catch {};
                    }
                } else if (isAssumptionKind(ann.kind)) {
                    assumption_annotations.append(ann) catch {};
                }
            }

            self.solver.reset();
            for (assumption_annotations.items) |ann| {
                for (ann.extra_constraints) |cst| {
                    self.solver.assert(cst);
                }
                self.solver.assert(ann.condition);
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

            std.debug.print("verification: {s} [base] start\n", .{fn_name});
            timer.reset();
            const assumption_status = self.solver.check();
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
                    std.debug.print("verification: {s} [base] -> SAT ({d}ms)\n", .{ fn_name, elapsed_ms });
                },
                z3.Z3_L_FALSE => {
                    std.debug.print("verification: {s} [base] -> UNSAT ({d}ms)\n", .{ fn_name, elapsed_ms });
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
                    std.debug.print("verification: {s} [base] -> UNKNOWN ({d}ms)\n", .{ fn_name, elapsed_ms });
                    std.debug.print("note: Z3 returned UNKNOWN (likely timeout). Keeping runtime guards.\n", .{});
                    // Treat UNKNOWN as non-fatal; leave guards in place.
                },
            }

            // Obligation proving: assumptions ∧ ¬obligation must be UNSAT.
            for (obligation_annotations.items) |ann| {
                self.solver.push();
                for (ann.path_constraints) |cst| {
                    self.solver.assert(cst);
                }
                for (ann.extra_constraints) |cst| {
                    self.solver.assert(cst);
                }
                const negated = z3.Z3_mk_not(self.context.ctx, ann.condition);
                self.solver.assert(negated);

                const obligation_label = obligationKindLabel(ann.kind);
                std.debug.print("verification: {s} [{s}] start\n", .{ fn_name, obligation_label });
                timer.reset();
                const obligation_status = self.solver.check();
                const obligation_ms = timer.read() / std.time.ns_per_ms;
                stats_total_queries += 1;
                stats_total_ms += obligation_ms;
                switch (obligation_status) {
                    z3.Z3_L_TRUE => stats_sat += 1,
                    z3.Z3_L_FALSE => stats_unsat += 1,
                    else => stats_unknown += 1,
                }
                std.debug.print("verification: {s} [{s}] -> {s} ({d}ms)\n", .{
                    fn_name,
                    obligation_label,
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
                    self.solver.pop();
                    return result;
                }
                if (obligation_status == z3.Z3_L_UNDEF) {
                    std.debug.print("note: Z3 returned UNKNOWN while proving {s} in {s}.\n", .{ obligation_label, fn_name });
                }

                self.solver.pop();

                if (ann.kind == .LoopInvariant) {
                    if (ann.old_condition) |old_inv| {
                        self.solver.push();
                        for (ann.path_constraints) |cst| {
                            self.solver.assert(cst);
                        }
                        for (ann.extra_constraints) |cst| {
                            self.solver.assert(cst);
                        }
                        for (ann.old_extra_constraints) |cst| {
                            self.solver.assert(cst);
                        }
                        for (ann.loop_step_extra_constraints) |cst| {
                            self.solver.assert(cst);
                        }
                        self.solver.assert(old_inv);
                        if (ann.loop_step_condition) |step_cond| {
                            self.solver.assert(step_cond);
                        }
                        self.solver.assert(z3.Z3_mk_not(self.context.ctx, ann.condition));

                        std.debug.print("verification: {s} [invariant-step] start\n", .{fn_name});
                        timer.reset();
                        const step_status = self.solver.check();
                        const step_ms = timer.read() / std.time.ns_per_ms;
                        stats_total_queries += 1;
                        stats_total_ms += step_ms;
                        switch (step_status) {
                            z3.Z3_L_TRUE => stats_sat += 1,
                            z3.Z3_L_FALSE => stats_unsat += 1,
                            else => stats_unknown += 1,
                        }
                        std.debug.print("verification: {s} [invariant-step] -> {s} ({d}ms)\n", .{
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
                            self.solver.pop();
                            return result;
                        }
                        if (step_status == z3.Z3_L_UNDEF) {
                            std.debug.print("note: Z3 returned UNKNOWN while proving invariant step in {s}.\n", .{fn_name});
                        }

                        self.solver.pop();
                    }
                }
            }

            // Explicit loop-post proving: assumptions ∧ invariant ∧ exit_condition ∧ ¬ensures must be UNSAT.
            for (loop_post_invariant_annotations.items) |inv_ann| {
                const exit_condition = inv_ann.loop_exit_condition orelse continue;
                for (ensure_annotations.items) |ensure_ann| {
                    self.solver.push();
                    for (inv_ann.path_constraints) |cst| {
                        self.solver.assert(cst);
                    }
                    for (ensure_ann.path_constraints) |cst| {
                        self.solver.assert(cst);
                    }
                    for (inv_ann.extra_constraints) |cst| {
                        self.solver.assert(cst);
                    }
                    for (inv_ann.loop_exit_extra_constraints) |cst| {
                        self.solver.assert(cst);
                    }
                    for (ensure_ann.extra_constraints) |cst| {
                        self.solver.assert(cst);
                    }
                    self.solver.assert(inv_ann.condition);
                    self.solver.assert(exit_condition);
                    self.solver.assert(z3.Z3_mk_not(self.context.ctx, ensure_ann.condition));

                    std.debug.print("verification: {s} [invariant-post] start\n", .{fn_name});
                    timer.reset();
                    const post_status = self.solver.check();
                    const post_ms = timer.read() / std.time.ns_per_ms;
                    stats_total_queries += 1;
                    stats_total_ms += post_ms;
                    switch (post_status) {
                        z3.Z3_L_TRUE => stats_sat += 1,
                        z3.Z3_L_FALSE => stats_unsat += 1,
                        else => stats_unknown += 1,
                    }
                    std.debug.print("verification: {s} [invariant-post] -> {s} ({d}ms)\n", .{
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
                        self.solver.pop();
                        return result;
                    }
                    if (post_status == z3.Z3_L_UNDEF) {
                        std.debug.print(
                            "note: Z3 returned UNKNOWN while proving loop-post condition in {s}.\n",
                            .{fn_name},
                        );
                    }

                    self.solver.pop();
                }
            }

            // Process refinement guards incrementally:
            // Each guard is checked in the context of all PREVIOUS guards (as assumptions)
            // This allows us to detect when a later guard is unsatisfiable given earlier constraints
            for (guard_annotations.items) |ann| {
                if (ann.guard_id == null) continue;

                // First check: can the guard EVER be satisfied given previous guards?
                // If (assumptions AND previous_guards AND this_guard) is UNSAT, error!
                self.solver.push();
                for (ann.path_constraints) |cst| {
                    self.solver.assert(cst);
                }
                for (ann.extra_constraints) |cst| {
                    self.solver.assert(cst);
                }
                self.solver.assert(ann.condition);
                std.debug.print("verification: {s} guard {s} [satisfy] start\n", .{ fn_name, ann.guard_id.? });
                timer.reset();
                const satisfy_status = self.solver.check();
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
                self.solver.pop();

                if (satisfy_status == z3.Z3_L_FALSE) {
                    // Guard can NEVER be satisfied given previous constraints - error!
                    std.debug.print("verification: {s} guard {s} [satisfy] -> UNSATISFIABLE ({d}ms)\n", .{ fn_name, ann.guard_id.?, satisfy_ms });
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
                    std.debug.print("verification: {s} guard {s} [satisfy] -> SAT ({d}ms)\n", .{ fn_name, ann.guard_id.?, satisfy_ms });
                } else {
                    std.debug.print("verification: {s} guard {s} [satisfy] -> UNKNOWN ({d}ms)\n", .{ fn_name, ann.guard_id.?, satisfy_ms });
                    std.debug.print("note: Z3 returned UNKNOWN (likely timeout). Keeping runtime guards.\n", .{});
                }

                // Second check: can the guard be violated? (to determine if it should be kept)
                self.solver.push();
                for (ann.path_constraints) |cst| {
                    self.solver.assert(cst);
                }
                for (ann.extra_constraints) |cst| {
                    self.solver.assert(cst);
                }
                const not_guard = z3.Z3_mk_not(self.context.ctx, ann.condition);
                self.solver.assert(not_guard);
                if (self.debug_z3) {
                    std.debug.print("[Z3] guard {s}\n", .{ann.guard_id.?});
                    self.logAst("guard", ann.condition);
                    self.logSolverState("guard");
                }
                std.debug.print("verification: {s} guard {s} [violate] start\n", .{ fn_name, ann.guard_id.? });
                timer.reset();
                const guard_status = self.solver.check();
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
                std.debug.print("verification: {s} guard {s} [violate] -> {s} ({d}ms)\n", .{
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
                }
                self.solver.pop();

                // Add this guard to context for subsequent guards
                // This builds up the constraint context incrementally
                for (ann.extra_constraints) |cst| {
                    self.solver.assert(cst);
                }
                for (ann.path_constraints) |cst| {
                    self.solver.assert(cst);
                }
                self.solver.assert(ann.condition);
            }
        }

        if (self.verify_stats) {
            std.debug.print(
                "verification stats: queries={d} sat={d} unsat={d} unknown={d} total_ms={d}\n",
                .{ stats_total_queries, stats_sat, stats_unsat, stats_unknown, stats_total_ms },
            );
        }

        return result;
    }

    fn runVerificationPassParallel(self: *VerificationPass, mlir_module: mlir.MlirModule) !errors.VerificationResult {
        try self.extractAnnotationsFromMLIR(mlir_module);
        var queries = try self.buildPreparedQueries();
        defer {
            for (queries.items) |*query| {
                query.deinit(self.allocator);
            }
            queries.deinit();
        }

        if (queries.items.len == 0) {
            return errors.VerificationResult.init(self.allocator);
        }

        var thread_safe_allocator = std.heap.ThreadSafeAllocator{ .child_allocator = std.heap.page_allocator };
        const worker_allocator = thread_safe_allocator.allocator();

        const total_queries = queries.items.len;
        const worker_count = @min(self.max_workers, total_queries);

        const QueryResult = struct {
            status: z3.Z3_lbool = z3.Z3_L_UNDEF,
            elapsed_ms: u64 = 0,
            err: ?anyerror = null,
            model_str: ?[]const u8 = null, // captured for SAT counterexample-capable queries
        };

        const results = try worker_allocator.alloc(QueryResult, total_queries);
        for (results) |*entry| {
            entry.* = .{};
        }
        defer worker_allocator.free(results);

        const WorkState = struct {
            queries: []const PreparedQuery,
            next_index: std.atomic.Value(usize),
            results: []QueryResult,
            allocator: std.mem.Allocator,
            timeout_ms: ?u32,
        };

        var state = WorkState{
            .queries = queries.items,
            .next_index = std.atomic.Value(usize).init(0),
            .results = results,
            .allocator = worker_allocator,
            .timeout_ms = self.timeout_ms,
        };

        const Worker = struct {
            fn run(ctx: *WorkState) void {
                var context = Context.init(ctx.allocator) catch return;
                defer context.deinit();

                var solver = Solver.init(&context, ctx.allocator) catch return;
                defer solver.deinit();

                if (ctx.timeout_ms) |ms| {
                    solver.setTimeoutMs(ms);
                }

                while (true) {
                    const idx = ctx.next_index.fetchAdd(1, .seq_cst);
                    if (idx >= ctx.queries.len) break;

                    const query = ctx.queries[idx];
                    solver.reset();
                    solver.loadFromSmtlib(query.smtlib_z);

                    std.debug.print("{s} start\n", .{query.log_prefix});
                    var timer = std.time.Timer.start() catch |err| {
                        ctx.results[idx].err = err;
                        continue;
                    };
                    const status = solver.check();
                    const elapsed_ms = timer.read() / std.time.ns_per_ms;

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
                        if (solver.getModel()) |model| {
                            const raw = z3.Z3_model_to_string(context.ctx, model);
                            if (raw != null) {
                                const dup = ctx.allocator.dupe(u8, std.mem.span(raw)) catch null;
                                ctx.results[idx].model_str = dup;
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

        var combined = errors.VerificationResult.init(self.allocator);
        var inconsistent_functions = std.StringHashMap(void).init(self.allocator);
        defer {
            var fn_it = inconsistent_functions.iterator();
            while (fn_it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            inconsistent_functions.deinit();
        }

        // First pass: surface base inconsistency and record affected functions.
        for (results, 0..) |entry, idx| {
            if (entry.err) |err| return err;
            const query = queries.items[idx];
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
                std.debug.print("note: Z3 returned UNKNOWN (likely timeout). Keeping runtime guards.\n", .{});
                // Treat UNKNOWN as non-fatal; leave guards in place.
            }
        }

        // Second pass: process all non-base queries, skipping inconsistent bases.
        for (results, 0..) |entry, idx| {
            const query = queries.items[idx];
            if (query.kind == .Base) continue;
            if (inconsistent_functions.contains(query.function_name)) continue;

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
                        std.debug.print(
                            "note: Z3 returned UNKNOWN while proving {s} in {s}.\n",
                            .{ obligationKindLabel(query.obligation_kind orelse .Ensures), query.function_name },
                        );
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
                        std.debug.print(
                            "note: Z3 returned UNKNOWN while proving invariant step in {s}.\n",
                            .{query.function_name},
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
                        std.debug.print(
                            "note: Z3 returned UNKNOWN while proving loop-post condition in {s}.\n",
                            .{query.function_name},
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
                    }
                },
                .Base => {},
            }
        }

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

    fn buildPreparedQueries(self: *VerificationPass) !ManagedArrayList(PreparedQuery) {
        var queries = ManagedArrayList(PreparedQuery).init(self.allocator);

        var by_function = std.StringHashMap(ManagedArrayList(EncodedAnnotation)).init(self.allocator);
        defer {
            var it = by_function.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            by_function.deinit();
        }

        for (self.encoded_annotations.items) |ann| {
            const entry = try by_function.getOrPut(ann.function_name);
            if (!entry.found_existing) {
                entry.value_ptr.* = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            }
            try entry.value_ptr.append(ann);
        }

        var it = by_function.iterator();
        while (it.next()) |entry| {
            const fn_name = entry.key_ptr.*;
            const annotations = entry.value_ptr.items;
            if (annotations.len == 0) continue;

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
                    guard_annotations.append(ann) catch {};
                } else if (ann.kind == .PathAssume) {
                    path_assumption_annotations.append(ann) catch {};
                } else if (isObligationKind(ann.kind)) {
                    obligation_annotations.append(ann) catch {};
                    if (ann.kind == .Ensures) {
                        ensure_annotations.append(ann) catch {};
                    } else if (ann.kind == .LoopInvariant and ann.loop_exit_condition != null) {
                        loop_post_invariant_annotations.append(ann) catch {};
                    }
                } else if (isAssumptionKind(ann.kind)) {
                    assumption_annotations.append(ann) catch {};
                }
            }

            var assumption_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
            defer assumption_constraints.deinit();

            for (assumption_annotations.items) |ann| {
                try addConstraintSlice(&assumption_constraints, ann.extra_constraints);
                try assumption_constraints.append(ann.condition);
            }

            const base_smtlib = try buildSmtlibForConstraints(self.allocator, &self.solver, assumption_constraints.items);
            const base_log_prefix = try std.fmt.allocPrint(self.allocator, "verification: {s} [base]", .{fn_name});
            try queries.append(.{
                .kind = .Base,
                .function_name = fn_name,
                .file = self.firstLocationFile(annotations),
                .line = self.firstLocationLine(annotations),
                .column = self.firstLocationColumn(annotations),
                .smtlib_z = base_smtlib,
                .log_prefix = base_log_prefix,
            });

            for (obligation_annotations.items) |ann| {
                var obligation_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer obligation_constraints.deinit();
                try addConstraintSlice(&obligation_constraints, assumption_constraints.items);
                try addConstraintSlice(&obligation_constraints, ann.path_constraints);
                try addConstraintSlice(&obligation_constraints, ann.extra_constraints);
                const negated = z3.Z3_mk_not(self.context.ctx, ann.condition);
                try obligation_constraints.append(negated);

                const obligation_smtlib = try buildSmtlibForConstraints(self.allocator, &self.solver, obligation_constraints.items);
                const obligation_log_prefix = try std.fmt.allocPrint(
                    self.allocator,
                    "verification: {s} [{s}]",
                    .{ fn_name, obligationKindLabel(ann.kind) },
                );
                try queries.append(.{
                    .kind = .Obligation,
                    .function_name = fn_name,
                    .obligation_kind = ann.kind,
                    .file = ann.file,
                    .line = ann.line,
                    .column = ann.column,
                    .smtlib_z = obligation_smtlib,
                    .log_prefix = obligation_log_prefix,
                });

                if (ann.kind == .LoopInvariant) {
                    if (ann.old_condition) |old_inv| {
                        var step_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                        defer step_constraints.deinit();
                        try addConstraintSlice(&step_constraints, assumption_constraints.items);
                        try addConstraintSlice(&step_constraints, ann.path_constraints);
                        try addConstraintSlice(&step_constraints, ann.extra_constraints);
                        try addConstraintSlice(&step_constraints, ann.old_extra_constraints);
                        try addConstraintSlice(&step_constraints, ann.loop_step_extra_constraints);
                        try step_constraints.append(old_inv);
                        if (ann.loop_step_condition) |step_cond| {
                            try step_constraints.append(step_cond);
                        }
                        try step_constraints.append(z3.Z3_mk_not(self.context.ctx, ann.condition));

                        const step_smtlib = try buildSmtlibForConstraints(self.allocator, &self.solver, step_constraints.items);
                        const step_log_prefix = try std.fmt.allocPrint(
                            self.allocator,
                            "verification: {s} [invariant-step]",
                            .{fn_name},
                        );
                        try queries.append(.{
                            .kind = .LoopInvariantStep,
                            .function_name = fn_name,
                            .obligation_kind = .LoopInvariant,
                            .file = ann.file,
                            .line = ann.line,
                            .column = ann.column,
                            .smtlib_z = step_smtlib,
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
                    try addConstraintSlice(&post_constraints, assumption_constraints.items);
                    try addConstraintSlice(&post_constraints, inv_ann.path_constraints);
                    try addConstraintSlice(&post_constraints, ensure_ann.path_constraints);
                    try addConstraintSlice(&post_constraints, inv_ann.extra_constraints);
                    try addConstraintSlice(&post_constraints, inv_ann.loop_exit_extra_constraints);
                    try addConstraintSlice(&post_constraints, ensure_ann.extra_constraints);
                    try post_constraints.append(inv_ann.condition);
                    try post_constraints.append(exit_condition);
                    try post_constraints.append(z3.Z3_mk_not(self.context.ctx, ensure_ann.condition));

                    const post_smtlib = try buildSmtlibForConstraints(self.allocator, &self.solver, post_constraints.items);
                    const post_log_prefix = try std.fmt.allocPrint(
                        self.allocator,
                        "verification: {s} [invariant-post]",
                        .{fn_name},
                    );
                    try queries.append(.{
                        .kind = .LoopInvariantPost,
                        .function_name = fn_name,
                        .obligation_kind = .Ensures,
                        .file = ensure_ann.file,
                        .line = ensure_ann.line,
                        .column = ensure_ann.column,
                        .smtlib_z = post_smtlib,
                        .log_prefix = post_log_prefix,
                    });
                }
            }

            var previous_guards = ManagedArrayList(EncodedAnnotation).init(self.allocator);
            defer previous_guards.deinit();

            for (guard_annotations.items) |ann| {
                if (ann.guard_id == null) continue;

                var guard_base = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer guard_base.deinit();
                try addConstraintSlice(&guard_base, assumption_constraints.items);
                try addConstraintSlice(&guard_base, ann.path_constraints);

                for (previous_guards.items) |prev| {
                    try addConstraintSlice(&guard_base, prev.path_constraints);
                    try addConstraintSlice(&guard_base, prev.extra_constraints);
                    try guard_base.append(prev.condition);
                }

                // Satisfy query
                var satisfy_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer satisfy_constraints.deinit();
                try addConstraintSlice(&satisfy_constraints, guard_base.items);
                try addConstraintSlice(&satisfy_constraints, ann.extra_constraints);
                try satisfy_constraints.append(ann.condition);

                const satisfy_smtlib = try buildSmtlibForConstraints(self.allocator, &self.solver, satisfy_constraints.items);
                const satisfy_log_prefix = try std.fmt.allocPrint(
                    self.allocator,
                    "verification: {s} guard {s} [satisfy]",
                    .{ fn_name, ann.guard_id.? },
                );
                try queries.append(.{
                    .kind = .GuardSatisfy,
                    .function_name = fn_name,
                    .guard_id = ann.guard_id,
                    .file = ann.file,
                    .line = ann.line,
                    .column = ann.column,
                    .smtlib_z = satisfy_smtlib,
                    .log_prefix = satisfy_log_prefix,
                });

                // Violate query
                var violate_constraints = ManagedArrayList(z3.Z3_ast).init(self.allocator);
                defer violate_constraints.deinit();
                try addConstraintSlice(&violate_constraints, guard_base.items);
                try addConstraintSlice(&violate_constraints, ann.extra_constraints);
                const not_guard = z3.Z3_mk_not(self.context.ctx, ann.condition);
                try violate_constraints.append(not_guard);

                const violate_smtlib = try buildSmtlibForConstraints(self.allocator, &self.solver, violate_constraints.items);
                const violate_log_prefix = try std.fmt.allocPrint(
                    self.allocator,
                    "verification: {s} guard {s} [violate]",
                    .{ fn_name, ann.guard_id.? },
                );
                try queries.append(.{
                    .kind = .GuardViolate,
                    .function_name = fn_name,
                    .guard_id = ann.guard_id,
                    .file = ann.file,
                    .line = ann.line,
                    .column = ann.column,
                    .smtlib_z = violate_smtlib,
                    .log_prefix = violate_log_prefix,
                });

                try previous_guards.append(ann);
            }
        }

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

    fn buildCounterexample(self: *VerificationPass) ?errors.Counterexample {
        const model = self.solver.getModel() orelse return null;
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

            // Filter out internal/synthetic variables
            if (std.mem.startsWith(u8, name, "undef_")) continue;
            if (std.mem.startsWith(u8, name, "old_")) continue;
            if (std.mem.startsWith(u8, name, "__")) continue;

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

        // Filter internal variables
        if (std.mem.startsWith(u8, name, "undef_")) continue;
        if (std.mem.startsWith(u8, name, "old_")) continue;
        if (std.mem.startsWith(u8, name, "__")) continue;

        if (value.len == 0) continue;
        ce.addVariable(name, value) catch continue;
    }

    if (ce.variables.count() == 0) {
        ce.deinit();
        return null;
    }
    return ce;
}

const ActivePathAssume = struct {
    condition: z3.Z3_ast,
    extra_constraints: []const z3.Z3_ast,
};

const EncodedAnnotation = struct {
    function_name: []const u8,
    kind: VerificationAnnotation.AnnotationKind,
    condition: z3.Z3_ast,
    extra_constraints: []const z3.Z3_ast,
    path_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    old_condition: ?z3.Z3_ast = null,
    old_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    loop_step_condition: ?z3.Z3_ast = null,
    loop_step_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    loop_exit_condition: ?z3.Z3_ast = null,
    loop_exit_extra_constraints: []const z3.Z3_ast = &[_]z3.Z3_ast{},
    file: []const u8,
    line: u32,
    column: u32,
    guard_id: ?[]const u8 = null,
};

fn isAssumptionKind(kind: VerificationAnnotation.AnnotationKind) bool {
    return switch (kind) {
        .Requires, .Assume => true,
        else => false,
    };
}

fn isObligationKind(kind: VerificationAnnotation.AnnotationKind) bool {
    return switch (kind) {
        .Ensures, .LoopInvariant, .ContractInvariant => true,
        else => false,
    };
}

fn obligationErrorType(kind: VerificationAnnotation.AnnotationKind) errors.VerificationErrorType {
    return switch (kind) {
        .Ensures => .PostconditionViolation,
        .LoopInvariant, .ContractInvariant => .InvariantViolation,
        else => .Unknown,
    };
}

fn obligationKindLabel(kind: VerificationAnnotation.AnnotationKind) []const u8 {
    return switch (kind) {
        .Ensures => "ensures",
        .LoopInvariant => "invariant",
        .ContractInvariant => "contract invariant",
        else => "obligation",
    };
}

const QueryKind = enum {
    Base,
    Obligation,
    LoopInvariantStep,
    LoopInvariantPost,
    GuardSatisfy,
    GuardViolate,
};

const PreparedQuery = struct {
    kind: QueryKind,
    function_name: []const u8,
    guard_id: ?[]const u8 = null,
    obligation_kind: ?VerificationAnnotation.AnnotationKind = null,
    file: []const u8,
    line: u32,
    column: u32,
    smtlib_z: [:0]const u8,
    log_prefix: []const u8,

    fn deinit(self: *PreparedQuery, allocator: std.mem.Allocator) void {
        allocator.free(self.smtlib_z);
        allocator.free(self.log_prefix);
    }
};

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

fn buildSmtlibForConstraints(
    allocator: std.mem.Allocator,
    solver: *Solver,
    constraints: []const z3.Z3_ast,
) ![:0]const u8 {
    solver.reset();
    for (constraints) |cst| {
        solver.assert(cst);
    }
    const raw = z3.Z3_solver_to_string(solver.context.ctx, solver.solver);
    const smtlib = if (raw == null) "" else std.mem.span(raw);
    return try allocator.dupeZ(u8, smtlib);
}

fn parseLocationString(loc: []const u8) struct { file: []const u8, line: u32, column: u32 } {
    const last = std.mem.lastIndexOfScalar(u8, loc, ':') orelse return .{ .file = "", .line = 0, .column = 0 };
    const before_last = loc[0..last];
    const second_last = std.mem.lastIndexOfScalar(u8, before_last, ':') orelse return .{ .file = "", .line = 0, .column = 0 };

    const file = before_last[0..second_last];
    const line_str = before_last[second_last + 1 ..];
    const col_str = loc[last + 1 ..];

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
        pass.solver.loadFromSmtlib(q.smtlib_z);
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
        pass.solver.loadFromSmtlib(q.smtlib_z);
        const status = pass.solver.check();
        try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), status);
    }
    try testing.expect(found_obligation);
}
