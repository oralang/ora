// ============================================================================
// MLIR Pass Manager
// ============================================================================
//
// Configures and runs optimization passes on MLIR modules.
//
// FEATURES:
//   • Configurable optimization pipeline
//   • Verification passes
//   • Canonicalization and CSE
//   • SCCP (Sparse Conditional Constant Propagation)
//   • Symbol DCE (Dead Code Elimination)
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
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

    /// Enable verification on this pass manager
    pub fn enableVerifier(self: *PassManager, enable: bool) void {
        c.mlirPassManagerEnableVerifier(self.pass_manager, enable);
    }

    /// Enable timing on this pass manager
    pub fn enableTiming(self: *PassManager) void {
        c.mlirPassManagerEnableTiming(self.pass_manager);
    }

    pub fn deinit(self: *PassManager) void {
        c.mlirPassManagerDestroy(self.pass_manager);
    }

    /// Add standard MLIR optimization passes
    pub fn addStandardOptimizationPasses(self: *PassManager) void {
        // use comprehensive optimization pipeline
        const pipeline_str = "builtin.module(canonicalize,cse,sccp,symbol-dce,mem2reg,loop-invariant-code-motion,cf-cfg-simplification)";
        const pipeline_ref = c.mlirStringRefCreateFromCString(pipeline_str);

        // parse and add the pipeline
        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(self.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("WARNING: Failed to parse standard optimization pipeline\n", .{});
        }
    }

    /// Add Ora-specific verification passes
    pub fn addOraVerificationPasses(self: *PassManager) void {
        // note: Ora-specific verification passes are not yet implemented
        // the MLIR built-in verifier (enabled via enableVerifier) is sufficient for now
        // when we implement custom verification passes, they would be added here
        _ = self;
    }

    /// Add custom passes from a pipeline string
    pub fn addCustomPasses(self: *PassManager, pipeline_str: []const u8) !void {
        const pipeline_ref = c.mlirStringRefCreate(pipeline_str.ptr, pipeline_str.len);
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
        const pipeline_ref = c.mlirStringRefCreate(pipeline_str.ptr, pipeline_str.len);
        const result = c.mlirParsePassPipeline(c.mlirPassManagerGetAsOpPassManager(pass_manager.pass_manager), pipeline_ref, null, null);

        if (c.mlirLogicalResultIsFailure(result)) {
            return error.FailedToParsePipeline;
        }
    }

    /// Create a pass manager with standard Ora passes
    pub fn createOraPassManager(ctx: c.MlirContext, allocator: std.mem.Allocator, config: PassPipelineConfig) !PassManager {
        var pass_manager = PassManager.init(ctx, allocator);

        // enable timing if requested
        if (config.enable_timing) {
            pass_manager.enableTiming();
        }

        // note: MLIR verification is enabled by default in PassManager
        // the --mlir-verify flag is documented but verification happens automatically
        // explicitly enabling it with enableVerifier() causes crashes with unregistered operations
        // todo: Investigate proper verification of custom dialect operations

        // add optimization passes based on level
        switch (config.optimization_level) {
            .None => {
                // only add verification if enabled
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

        // add custom passes if provided
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

    // convert PipelineConfig to PassPipelineConfig
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
    };

    // create pass manager
    var pass_manager = try OraPassUtils.createOraPassManager(ctx, allocator, pass_config);
    defer pass_manager.deinit();

    // run passes
    const success = try pass_manager.run(module);
    result.success = success;

    if (!success) {
        result.error_message = try std.fmt.allocPrint(allocator, "MLIR pipeline failed", .{});
    }

    return result;
}
