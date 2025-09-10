const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const verification = @import("verification.zig");
const optimization = @import("optimization.zig");

/// MLIR pass integration and management system
pub const PassManager = struct {
    ctx: c.MlirContext,
    pass_manager: c.MlirPassManager,
    allocator: std.mem.Allocator,

    pub fn init(ctx: c.MlirContext, allocator: std.mem.Allocator) PassManager {
        const pass_manager = c.mlirPassManagerCreate(ctx);
        return .{
            .ctx = ctx,
            .pass_manager = pass_manager,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PassManager) void {
        c.mlirPassManagerDestroy(self.pass_manager);
    }

    /// Add standard MLIR optimization passes
    pub fn addStandardOptimizationPasses(self: *PassManager) void {
        // Use comprehensive optimization pipeline
        const pipeline_str = "builtin.module(canonicalize,cse,sccp,symbol-dce,mem2reg,loop-invariant-code-motion,cf-cfg-simplification)";
        const pipeline_ref = c.mlirStringRefCreateFromCString(pipeline_str);

        // Parse and add the pipeline
        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(self.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("WARNING: Failed to parse standard optimization pipeline\n", .{});
        }
    }

    /// Add Ora-specific verification passes
    pub fn addOraVerificationPasses(self: *PassManager) void {
        // Add standard MLIR verification pass first
        // Note: mlirCreateVerifierPass might not be available in all MLIR versions
        // For now, we'll use pipeline string parsing instead

        // Add Ora-specific verification passes using pipeline strings
        // These will be implemented as custom verification logic
        const ora_verification_pipeline = "builtin.module(ora-type-verify,ora-memory-verify,ora-contract-verify)";
        const pipeline_ref = c.mlirStringRefCreateFromCString(ora_verification_pipeline);

        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(self.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("WARNING: Failed to parse Ora verification pipeline - using fallback verification\n", .{});
            // Fallback to basic verification
            self.addBasicOraVerification();
        }
    }

    /// Add basic Ora verification using available MLIR passes
    fn addBasicOraVerification(self: *PassManager) void {
        // Use standard MLIR verification passes that work with unregistered dialects
        const basic_verification_pipeline = "builtin.module(verify-dominance,verify-loopinfo)";
        const pipeline_ref = c.mlirStringRefCreateFromCString(basic_verification_pipeline);

        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(self.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("WARNING: Failed to parse basic verification pipeline\n", .{});
        }
    }

    /// Add arithmetic optimization passes
    pub fn addArithmeticOptimizationPasses(self: *PassManager) void {
        // Use pipeline string parsing for arithmetic passes
        const pipeline_str = "builtin.module(arith-canonicalize,arith-expand-ops)";
        const pipeline_ref = c.mlirStringRefCreateFromCString(pipeline_str);

        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(self.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("WARNING: Failed to parse arithmetic optimization pipeline\n", .{});
        }
    }

    /// Add control flow optimization passes
    pub fn addControlFlowOptimizationPasses(self: *PassManager) void {
        // Use pipeline string parsing for control flow passes
        const pipeline_str = "builtin.module(scf-canonicalize,loop-invariant-code-motion)";
        const pipeline_ref = c.mlirStringRefCreateFromCString(pipeline_str);

        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(self.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("WARNING: Failed to parse control flow optimization pipeline\n", .{});
        }
    }

    /// Configure pass pipeline based on optimization level
    pub fn configurePipeline(self: *PassManager, config: PassPipelineConfig) void {
        switch (config.optimization_level) {
            .None => {
                // Only add verification passes for debug builds
                if (config.enable_verification) {
                    self.addOraVerificationPasses();
                }
            },
            .Basic => {
                // Add basic optimization passes
                self.addStandardOptimizationPasses();
                if (config.enable_verification) {
                    self.addOraVerificationPasses();
                }
            },
            .Aggressive => {
                // Add all optimization passes
                self.addStandardOptimizationPasses();
                self.addArithmeticOptimizationPasses();
                self.addControlFlowOptimizationPasses();
                if (config.enable_verification) {
                    self.addOraVerificationPasses();
                }
            },
        }

        // Add custom passes (placeholder - would need external pass API)
        for (config.custom_passes) |pass_name| {
            std.debug.print("WARNING: Custom pass '{s}' not yet implemented\n", .{pass_name});
        }
    }

    /// Run the configured pass pipeline on a module
    pub fn runPasses(self: *PassManager, module: c.MlirModule) !PassResult {
        const result = c.mlirPassManagerRunOnOp(self.pass_manager, c.mlirModuleGetOperation(module));

        if (c.mlirLogicalResultIsFailure(result)) {
            return PassResult{
                .success = false,
                .error_message = "Pass pipeline execution failed",
                .modified_module = module,
            };
        }

        return PassResult{
            .success = true,
            .error_message = null,
            .modified_module = module,
        };
    }

    // Custom pass creation functions removed - would need to be implemented using external pass API

    /// Enable pass timing and statistics
    pub fn enableTiming(self: *PassManager) void {
        c.mlirPassManagerEnableTiming(self.pass_manager);
    }

    /// Enable IR printing before and after passes
    pub fn enableIRPrinting(self: *PassManager, config: IRPrintingConfig) void {
        if (config.print_before_all) {
            c.mlirPassManagerEnableIRPrinting(self.pass_manager, true, false, false, false, false, c.MlirOpPrintingFlags{ .ptr = null }, c.MlirStringRef{ .data = null, .length = 0 });
        }
        if (config.print_after_all) {
            c.mlirPassManagerEnableIRPrinting(self.pass_manager, false, true, false, false, false, c.MlirOpPrintingFlags{ .ptr = null }, c.MlirStringRef{ .data = null, .length = 0 });
        }
        if (config.print_after_change) {
            c.mlirPassManagerEnableIRPrinting(self.pass_manager, false, false, true, false, false, c.MlirOpPrintingFlags{ .ptr = null }, c.MlirStringRef{ .data = null, .length = 0 });
        }
        if (config.print_after_failure) {
            c.mlirPassManagerEnableIRPrinting(self.pass_manager, false, false, false, true, false, c.MlirOpPrintingFlags{ .ptr = null }, c.MlirStringRef{ .data = null, .length = 0 });
        }
    }

    /// Verify the module after running passes
    pub fn verifyModule(self: *PassManager, module: c.MlirModule) bool {
        _ = self;
        return c.mlirOperationVerify(c.mlirModuleGetOperation(module));
    }

    /// Run Ora-specific verification on a module
    pub fn runOraVerification(self: *PassManager, module: c.MlirModule) !verification.VerificationResult {
        return try verification.runOraVerification(self.ctx, module, self.allocator);
    }

    /// Run Ora-specific optimizations on a module
    pub fn runOraOptimization(self: *PassManager, module: c.MlirModule, config: optimization.OptimizationConfig) !optimization.OptimizationResult {
        return try optimization.runOraOptimization(self.ctx, module, self.allocator, config);
    }
};

/// Pass pipeline configuration
pub const PassPipelineConfig = struct {
    optimization_level: OptimizationLevel,
    enable_verification: bool,
    custom_passes: []const []const u8,
    enable_timing: bool,
    ir_printing: IRPrintingConfig,

    pub fn default() PassPipelineConfig {
        return .{
            .optimization_level = .Basic,
            .enable_verification = true,
            .custom_passes = &[_][]const u8{},
            .enable_timing = false,
            .ir_printing = IRPrintingConfig.default(),
        };
    }

    pub fn debug() PassPipelineConfig {
        return .{
            .optimization_level = .None,
            .enable_verification = true,
            .custom_passes = &[_][]const u8{ "ora-memory-verify", "ora-type-verify" },
            .enable_timing = true,
            .ir_printing = IRPrintingConfig{
                .print_before_all = true,
                .print_after_all = true,
                .print_after_change = true,
                .print_after_failure = true,
            },
        };
    }

    pub fn release() PassPipelineConfig {
        return .{
            .optimization_level = .Aggressive,
            .enable_verification = false,
            .custom_passes = &[_][]const u8{},
            .enable_timing = false,
            .ir_printing = IRPrintingConfig.default(),
        };
    }
};

/// Optimization levels
pub const OptimizationLevel = enum {
    None, // No optimization, only verification
    Basic, // Basic optimizations (canonicalization, CSE, etc.)
    Aggressive, // All available optimizations
};

/// IR printing configuration
pub const IRPrintingConfig = struct {
    print_before_all: bool,
    print_after_all: bool,
    print_after_change: bool,
    print_after_failure: bool,

    pub fn default() IRPrintingConfig {
        return .{
            .print_before_all = false,
            .print_after_all = false,
            .print_after_change = false,
            .print_after_failure = true,
        };
    }
};

/// Result of running passes
pub const PassResult = struct {
    success: bool,
    error_message: ?[]const u8,
    modified_module: c.MlirModule,
};

/// Ora-specific pass utilities
pub const OraPassUtils = struct {
    /// Create a pass pipeline string for command-line usage
    pub fn createPipelineString(config: PassPipelineConfig, allocator: std.mem.Allocator) ![]u8 {
        var pipeline = std.ArrayList(u8).init(allocator);
        defer pipeline.deinit();

        try pipeline.appendSlice("builtin.module(");

        // Add optimization passes based on level
        switch (config.optimization_level) {
            .None => {
                if (config.enable_verification) {
                    try pipeline.appendSlice("ora-memory-verify,ora-type-verify");
                }
            },
            .Basic => {
                try pipeline.appendSlice("canonicalize,cse,sccp");
                if (config.enable_verification) {
                    try pipeline.appendSlice(",ora-memory-verify,ora-type-verify");
                }
            },
            .Aggressive => {
                try pipeline.appendSlice("canonicalize,cse,sccp,symbol-dce,arith-canonicalize,scf-canonicalize");
                if (config.enable_verification) {
                    try pipeline.appendSlice(",ora-memory-verify,ora-type-verify,ora-invariant-verify");
                }
            },
        }

        // Add custom passes
        for (config.custom_passes) |pass_name| {
            try pipeline.appendSlice(",");
            try pipeline.appendSlice(pass_name);
        }

        try pipeline.appendSlice(")");

        return pipeline.toOwnedSlice();
    }

    /// Parse pass pipeline string and configure pass manager
    pub fn parsePipelineString(pass_manager: *PassManager, pipeline_str: []const u8) !void {
        // Use the MLIR C API to parse the pipeline string
        const pipeline_ref = c.mlirStringRefCreate(pipeline_str.ptr, pipeline_str.len);

        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(pass_manager.pass_manager), pipeline_ref, null, // No error callback for now
            null);

        if (c.mlirLogicalResultIsFailure(result)) {
            return error.PipelineParsingFailed;
        }
    }

    /// Get available pass names
    pub fn getAvailablePassNames() []const []const u8 {
        return &[_][]const u8{
            // Standard MLIR passes
            "canonicalize",
            "cse",
            "sccp",
            "symbol-dce",
            "arith-canonicalize",
            "arith-expand-ops",
            "scf-canonicalize",
            "loop-invariant-code-motion",

            // Ora-specific passes
            "ora-memory-verify",
            "ora-type-verify",
            "ora-invariant-verify",
        };
    }

    /// Validate pass name
    pub fn isValidPassName(pass_name: []const u8) bool {
        const available_passes = getAvailablePassNames();
        for (available_passes) |available_pass| {
            if (std.mem.eql(u8, pass_name, available_pass)) {
                return true;
            }
        }
        return false;
    }
};
