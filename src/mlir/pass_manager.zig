const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const verification = @import("verification.zig");

/// MLIR Pipeline Configuration
pub const PipelineConfig = struct {
    /// Enable verification passes
    verify: bool = true,
    /// Enable canonicalization
    canonicalize: bool = true,
    /// Enable Common Subexpression Elimination
    cse: bool = true,
    /// Enable Sparse Conditional Constant Propagation
    sccp: bool = true,
    /// Enable memory-to-register optimization
    mem2reg: bool = true,
    /// Enable loop-invariant code motion
    licm: bool = true,
    /// Enable symbol dead code elimination
    symbol_dce: bool = false, // Disabled by default due to unregistered dialect issues
    /// Enable inlining (disabled by default due to unregistered dialect issues)
    @"inline": bool = false,
    /// Enable CFG simplification
    cfg_simplify: bool = false,
    /// Custom pass pipeline string (overrides individual flags)
    custom_pipeline: ?[]const u8 = null,
};

/// Default aggressive optimization pipeline
pub const aggressive_config = PipelineConfig{
    .verify = true,
    .canonicalize = true,
    .cse = true,
    .sccp = true,
    .mem2reg = true,
    .licm = true,
    .symbol_dce = false, // Disabled due to unregistered dialect
    .@"inline" = false, // Disabled due to unregistered dialect
    .cfg_simplify = false,
};

/// Default basic optimization pipeline
pub const basic_config = PipelineConfig{
    .verify = true,
    .canonicalize = true,
    .cse = true,
    .sccp = false,
    .mem2reg = true,
    .licm = false,
    .symbol_dce = false,
    .@"inline" = false,
    .cfg_simplify = false,
};

/// Default no optimization pipeline
pub const no_opt_config = PipelineConfig{
    .verify = true,
    .canonicalize = false,
    .cse = false,
    .sccp = false,
    .mem2reg = false,
    .licm = false,
    .symbol_dce = false,
    .@"inline" = false,
    .cfg_simplify = false,
    .custom_pipeline = "builtin.module()",
};

/// MLIR Pipeline Result
pub const PipelineResult = struct {
    success: bool,
    error_message: ?[]const u8,
    passes_applied: std.ArrayList([]const u8),
    optimization_level: OptimizationLevel,
    timing_info: ?[]const u8,
    modified_module: c.MlirModule,
    optimized_module: c.MlirModule,

    pub fn init(_: std.mem.Allocator) PipelineResult {
        return PipelineResult{
            .success = false,
            .error_message = null,
            .passes_applied = std.ArrayList([]const u8){},
            .optimization_level = .None,
            .timing_info = null,
            .modified_module = c.MlirModule{},
            .optimized_module = c.MlirModule{},
        };
    }

    pub fn deinit(self: *PipelineResult, allocator: std.mem.Allocator) void {
        self.passes_applied.deinit(allocator);
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
        if (self.timing_info) |timing| {
            allocator.free(timing);
        }
    }
};

/// Optimization levels
pub const OptimizationLevel = enum {
    None,
    Basic,
    Aggressive,
};

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
            std.debug.print("WARNING: Failed to parse Ora verification pipeline\n", .{});
        }
    }

    /// Add custom passes from a pipeline string
    pub fn addCustomPasses(self: *PassManager, pipeline_str: []const u8) !void {
        const pipeline_ref = c.mlirStringRefCreateFromCString(pipeline_str.ptr);
        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(self.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            return error.FailedToParsePipeline;
        }
    }

    /// Run the pass manager on a module
    pub fn run(self: *PassManager, module: c.MlirModule) !bool {
        const op = c.mlirModuleGetOperation(module);
        const result = c.mlirPassManagerRunOnOp(self.pass_manager, op);
        return c.mlirLogicalResultIsSuccess(result);
    }
};

/// Pass pipeline configuration for different optimization levels
pub const PassPipelineConfig = struct {
    optimization_level: OptimizationLevel,
    enable_verification: bool,
    custom_passes: []const []const u8,
    enable_timing: bool,
    ir_printing: IRPrintingConfig,
};

/// IR printing configuration
pub const IRPrintingConfig = struct {
    print_before_all: bool = false,
    print_after_all: bool = false,
    print_after_change: bool = false,
    print_after_failure: bool = true,
};

/// Pass result information
pub const PassResult = struct {
    success: bool,
    error_message: ?[]const u8,
    passes_run: std.ArrayList([]const u8),
    timing_info: ?[]const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PassResult {
        return PassResult{
            .success = false,
            .error_message = null,
            .passes_run = std.ArrayList([]const u8).init(allocator),
            .timing_info = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PassResult) void {
        self.passes_run.deinit(self.allocator);
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
        if (self.timing_info) |timing| {
            self.allocator.free(timing);
        }
    }
};

/// Ora-specific pass utilities
pub const OraPassUtils = struct {
    /// Parse a pipeline string and add passes to the pass manager
    pub fn parsePipelineString(pass_manager: *PassManager, pipeline_str: []const u8) !void {
        const pipeline_ref = c.mlirStringRefCreateFromCString(pipeline_str.ptr);
        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(pass_manager.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            return error.FailedToParsePipeline;
        }
    }

    /// Create a pass manager with standard Ora passes
    pub fn createOraPassManager(ctx: c.MlirContext, allocator: std.mem.Allocator, config: PassPipelineConfig) !PassManager {
        var pass_manager = PassManager.init(ctx, allocator);

        // Add verification passes if enabled
        if (config.enable_verification) {
            pass_manager.addOraVerificationPasses();
        }

        // Add optimization passes based on level
        switch (config.optimization_level) {
            .None => {
                // Only add verification if enabled
            },
            .Basic => {
                const basic_pipeline = "builtin.module(canonicalize,cse,mem2reg)";
                try pass_manager.addCustomPasses(basic_pipeline);
            },
            .Aggressive => {
                const aggressive_pipeline = "builtin.module(canonicalize,cse,sccp,mem2reg,loop-invariant-code-motion)";
                try pass_manager.addCustomPasses(aggressive_pipeline);
            },
        }

        // Add custom passes if provided
        for (config.custom_passes) |pass_name| {
            const pass_pipeline = try std.fmt.allocPrint(allocator, "builtin.module({s})", .{pass_name});
            defer allocator.free(pass_pipeline);
            try pass_manager.addCustomPasses(pass_pipeline);
        }

        return pass_manager;
    }
};

/// Run MLIR pipeline with configuration
pub fn runMLIRPipeline(ctx: c.MlirContext, module: c.MlirModule, config: PipelineConfig, allocator: std.mem.Allocator) !PipelineResult {
    var result = PipelineResult.init(allocator);
    defer result.deinit(allocator);

    // Convert PipelineConfig to PassPipelineConfig
    const pass_config = PassPipelineConfig{
        .optimization_level = if (config.custom_pipeline != null) .None else blk: {
            if (config.canonicalize and config.cse and config.sccp and config.mem2reg and config.licm) {
                break :blk .Aggressive;
            } else if (config.canonicalize and config.cse and config.mem2reg) {
                break :blk .Basic;
            } else {
                break :blk .None;
            }
        },
        .enable_verification = config.verify,
        .custom_passes = &[_][]const u8{},
        .enable_timing = false,
        .ir_printing = IRPrintingConfig{},
    };

    // Create pass manager
    var pass_manager = try OraPassUtils.createOraPassManager(ctx, allocator, pass_config);
    defer pass_manager.deinit();

    // Run passes
    const success = try pass_manager.run(module);
    result.success = success;

    if (!success) {
        result.error_message = try std.fmt.allocPrint(allocator, "MLIR pipeline failed", .{});
    }

    return result;
}
