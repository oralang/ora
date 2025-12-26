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
    context: Context,
    solver: Solver,
    encoder: Encoder,
    allocator: std.mem.Allocator,

    /// Map from function name to its annotations
    function_annotations: std.StringHashMap(FunctionAnnotations),

    /// List of loop invariants (with their locations)
    loop_invariants: ManagedArrayList(LoopInvariants),

    /// Current function being processed (for MLIR extraction)
    current_function_name: ?[]const u8 = null,

    /// Encoded annotations collected from MLIR
    encoded_annotations: ManagedArrayList(EncodedAnnotation),

    /// Storage for duplicated function names
    function_name_storage: ManagedArrayList([]const u8),
    /// Storage for duplicated location file names
    location_storage: ManagedArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator) !VerificationPass {
        var context = try Context.init(allocator);
        errdefer context.deinit();

        var solver = try Solver.init(&context, allocator);
        errdefer solver.deinit();

        const encoder = Encoder.init(&context, allocator);

        return VerificationPass{
            .context = context,
            .solver = solver,
            .encoder = encoder,
            .allocator = allocator,
            .function_annotations = std.StringHashMap(FunctionAnnotations).init(allocator),
            .loop_invariants = ManagedArrayList(LoopInvariants).init(allocator),
            .encoded_annotations = ManagedArrayList(EncodedAnnotation).init(allocator),
            .function_name_storage = ManagedArrayList([]const u8).init(allocator),
            .location_storage = ManagedArrayList([]const u8).init(allocator),
        };
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
        self.encoded_annotations.deinit();
        self.encoder.deinit();
        self.solver.deinit();
        self.context.deinit();
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

        // walk all regions in the module
        const num_regions = mlir.oraOperationGetNumRegions(module_op);
        for (0..@intCast(num_regions)) |region_idx| {
            const region = mlir.oraOperationGetRegion(module_op, @intCast(region_idx));
            try self.walkMLIRRegion(region);
        }
    }

    /// Walk an MLIR region to find verification operations
    fn walkMLIRRegion(self: *VerificationPass, region: mlir.MlirRegion) !void {
        // get first block in region
        var current_block = mlir.oraRegionGetFirstBlock(region);

        while (!mlir.oraBlockIsNull(current_block)) {
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

        // check for verification operations
        if (std.mem.eql(u8, op_name, "ora.requires")) {
            // extract requires condition
            // get the condition operand (should be the first and only operand)
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                try self.recordEncodedAnnotation(op, .Requires, condition_value);
            }
        } else if (std.mem.eql(u8, op_name, "ora.ensures")) {
            // extract ensures condition
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                try self.recordEncodedAnnotation(op, .Ensures, condition_value);
            }
        } else if (std.mem.eql(u8, op_name, "ora.invariant")) {
            // extract invariant condition
            const num_operands = mlir.oraOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = mlir.oraOperationGetOperand(op, 0);
                try self.recordEncodedAnnotation(op, .LoopInvariant, condition_value);
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

    fn recordEncodedAnnotation(
        self: *VerificationPass,
        op: mlir.MlirOperation,
        kind: VerificationAnnotation.AnnotationKind,
        condition_value: mlir.MlirValue,
    ) !void {
        const function_name = self.current_function_name orelse "unknown";
        const encoded = try self.encoder.encodeValue(condition_value);
        const loc = try self.getLocationInfo(op);
        try self.encoded_annotations.append(.{
            .function_name = function_name,
            .kind = kind,
            .condition = encoded,
            .file = loc.file,
            .line = loc.line,
            .column = loc.column,
        });
    }

    pub fn runVerificationPass(self: *VerificationPass, mlir_module: mlir.MlirModule) !errors.VerificationResult {
        try self.extractAnnotationsFromMLIR(mlir_module);
        var result = errors.VerificationResult.init(self.allocator);

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

            self.solver.reset();
            for (annotations) |ann| {
                self.solver.assert(ann.condition);
            }

            const status = self.solver.check();
            switch (status) {
                z3.Z3_L_TRUE => {
                    std.debug.print("verification: {s} -> SAT\n", .{fn_name});
                },
                z3.Z3_L_FALSE => {
                    std.debug.print("verification: {s} -> UNSAT\n", .{fn_name});
                    const counterexample = self.buildCounterexample();
                    try result.addError(.{
                        .error_type = .InvariantViolation,
                        .message = try std.fmt.allocPrint(self.allocator, "verification failed (UNSAT) in {s}", .{fn_name}),
                        .file = try self.allocator.dupe(u8, self.firstLocationFile(annotations)),
                        .line = self.firstLocationLine(annotations),
                        .column = self.firstLocationColumn(annotations),
                        .counterexample = counterexample,
                        .allocator = self.allocator,
                    });
                    return result;
                },
                else => {
                    std.debug.print("verification: {s} -> UNKNOWN\n", .{fn_name});
                    const counterexample = self.buildCounterexample();
                    try result.addError(.{
                        .error_type = .Unknown,
                        .message = try std.fmt.allocPrint(self.allocator, "verification result unknown in {s}", .{fn_name}),
                        .file = try self.allocator.dupe(u8, self.firstLocationFile(annotations)),
                        .line = self.firstLocationLine(annotations),
                        .column = self.firstLocationColumn(annotations),
                        .counterexample = counterexample,
                        .allocator = self.allocator,
                    });
                    return result;
                },
            }
        }

        return result;
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
        const model_str = z3.Z3_model_to_string(self.context.ctx, model);
        if (model_str == null) return null;

        var ce = errors.Counterexample.init(self.allocator);
        const model_slice = std.mem.span(model_str);
        const key = "__model";
        ce.addVariable(key, model_slice) catch {
            ce.deinit();
            return null;
        };
        return ce;
    }
};

const EncodedAnnotation = struct {
    function_name: []const u8,
    kind: VerificationAnnotation.AnnotationKind,
    condition: z3.Z3_ast,
    file: []const u8,
    line: u32,
    column: u32,
};

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
