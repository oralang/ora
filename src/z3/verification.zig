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
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const Context = @import("context.zig").Context;
const Solver = @import("solver.zig").Solver;
const Encoder = @import("encoder.zig").Encoder;
const errors = @import("errors.zig");

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
        // Note: We don't own the ExprNode pointers, they're from AST arena
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
    loop_invariants: std.ArrayList(LoopInvariants),

    /// Current function being processed (for MLIR extraction)
    current_function_name: ?[]const u8 = null,

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
            .loop_invariants = std.ArrayList(LoopInvariants).init(allocator),
        };
    }

    pub fn deinit(self: *VerificationPass) void {
        var iter = self.function_annotations.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.function_annotations.deinit();
        self.loop_invariants.deinit();
        self.encoder.deinit();
        self.solver.deinit();
        self.context.deinit();
    }

    //===----------------------------------------------------------------------===//
    // Annotation Extraction from AST
    //===----------------------------------------------------------------------===//

    /// Extract verification annotations from AST module
    pub fn extractAnnotationsFromAST(self: *VerificationPass, module: *lib.ast.AstNode) !void {
        switch (module.*) {
            .Module => |*mod| {
                // Walk all top-level declarations
                for (mod.declarations) |*decl| {
                    try self.extractAnnotationsFromDeclaration(decl);
                }
            },
            else => {
                // Not a module, try to extract from the node directly
                try self.extractAnnotationsFromDeclaration(module);
            },
        }
    }

    /// Extract annotations from a declaration (contract, function, etc.)
    fn extractAnnotationsFromDeclaration(self: *VerificationPass, decl: *lib.ast.AstNode) !void {
        switch (decl.*) {
            .Contract => |*contract| {
                // Extract annotations from contract members
                for (contract.body) |*member| {
                    try self.extractAnnotationsFromDeclaration(member);
                }
            },
            .Function => |*function| {
                // Include ghost functions in verification (they're specification-only)
                // Ghost functions are used for verification but not compiled to bytecode
                try self.extractFunctionAnnotations(function);
            },
            else => {
                // Other declaration types don't have verification annotations
            },
        }
    }

    /// Extract requires/ensures clauses from a function node
    fn extractFunctionAnnotations(self: *VerificationPass, function: *lib.ast.FunctionNode) !void {
        const function_name = function.name;

        // Create function annotations entry
        var annotations = FunctionAnnotations.init(self.allocator, function_name);

        // Extract requires clauses
        if (function.requires_clauses.len > 0) {
            annotations.requires = try self.allocator.dupe(*lib.ast.Expressions.ExprNode, function.requires_clauses);
        }

        // Extract ensures clauses
        if (function.ensures_clauses.len > 0) {
            annotations.ensures = try self.allocator.dupe(*lib.ast.Expressions.ExprNode, function.ensures_clauses);
        }

        // Store in map
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
    // Annotation Extraction from MLIR
    //===----------------------------------------------------------------------===//

    /// Extract verification annotations from MLIR module
    /// This walks MLIR operations looking for ora.requires, ora.ensures, ora.invariant
    pub fn extractAnnotationsFromMLIR(self: *VerificationPass, mlir_module: c.MlirModule) !void {
        // Get the module operation
        const module_op = c.mlirModuleGetOperation(mlir_module);

        // Walk all regions in the module
        const num_regions = c.mlirOperationGetNumRegions(module_op);
        for (0..@intCast(num_regions)) |region_idx| {
            const region = c.mlirOperationGetRegion(module_op, @intCast(region_idx));
            try self.walkMLIRRegion(region);
        }
    }

    /// Walk an MLIR region to find verification operations
    fn walkMLIRRegion(self: *VerificationPass, region: c.MlirRegion) !void {
        // Get first block in region
        var current_block = c.mlirRegionGetFirstBlock(region);

        while (!c.mlirBlockIsNull(current_block)) {
            // Walk operations in this block
            var current_op = c.mlirBlockGetFirstOperation(current_block);

            while (!c.mlirOperationIsNull(current_op)) {
                try self.processMLIROperation(current_op);

                // Walk nested regions (for functions, if statements, etc.)
                const num_regions = c.mlirOperationGetNumRegions(current_op);
                for (0..@intCast(num_regions)) |region_idx| {
                    const nested_region = c.mlirOperationGetRegion(current_op, @intCast(region_idx));
                    try self.walkMLIRRegion(nested_region);
                }

                current_op = c.mlirOperationGetNextInBlock(current_op);
            }

            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }

    /// Process a single MLIR operation to extract verification annotations
    fn processMLIROperation(self: *VerificationPass, op: c.MlirOperation) !void {
        const op_name = self.getMLIROperationName(op);

        // Track function entry to associate annotations with functions
        if (std.mem.eql(u8, op_name, "func.func")) {
            // Extract function name from attributes
            // TODO: Get function name from func.func operation
            // For now, we'll track it when we encounter requires/ensures
        }

        // Check for verification operations
        if (std.mem.eql(u8, op_name, "ora.requires")) {
            // Extract requires condition
            // Get the condition operand (should be the first and only operand)
            const num_operands = c.mlirOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = c.mlirOperationGetOperand(op, 0);
                _ = condition_value; // Will be used when encoding to Z3 (z3-2)
                // TODO: Encode condition_value to Z3 AST and store it
                // This will be completed when we implement VCG (z3-2)
            }
        } else if (std.mem.eql(u8, op_name, "ora.ensures")) {
            // Extract ensures condition
            const num_operands = c.mlirOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = c.mlirOperationGetOperand(op, 0);
                _ = condition_value; // Will be used when encoding to Z3 (z3-2)
                // TODO: Encode condition_value to Z3 AST and store it
            }
        } else if (std.mem.eql(u8, op_name, "ora.invariant")) {
            // Extract invariant condition
            const num_operands = c.mlirOperationGetNumOperands(op);
            if (num_operands >= 1) {
                const condition_value = c.mlirOperationGetOperand(op, 0);
                _ = condition_value; // Will be used when encoding to Z3 (z3-2)
                // TODO: Encode condition_value to Z3 AST and store it
            }
        }
    }

    /// Get MLIR operation name as string
    fn getMLIROperationName(_: *VerificationPass, op: c.MlirOperation) []const u8 {
        const op_name = c.mlirOperationGetName(op);
        const op_name_str = c.mlirIdentifierStr(op_name);
        // Create a slice from the MLIR string reference
        // Note: This is safe as long as the MLIR context is alive
        return op_name_str.data[0..op_name_str.length];
    }

    // TODO: Implement verification methods
    // - verifyArithmeticSafety
    // - verifyBounds
    // - verifyStorageConsistency
    // - verifyUserInvariants
    // - runVerificationPass
};
