const std = @import("std");
pub const ast = @import("../ast.zig");

/// Analysis state tracking for memory safety
pub const AnalysisState = struct {
    phase: AnalysisPhase,
    current_node_type: ?std.meta.Tag(ast.AstNode),
    error_count: u32,
    warning_count: u32,
    safety_checks_enabled: bool,

    pub const AnalysisPhase = enum {
        PreInitialization,
        TypeChecking,
        SemanticAnalysis,
        Validation,
    };

    pub fn init() AnalysisState {
        return AnalysisState{
            .phase = .PreInitialization,
            .current_node_type = null,
            .error_count = 0,
            .warning_count = 0,
            .safety_checks_enabled = true,
        };
    }
};

/// Validation coverage tracking
pub const ValidationCoverage = struct {
    visited_node_types: std.EnumSet(std.meta.Tag(ast.AstNode)),
    missing_implementations: std.ArrayList(std.meta.Tag(ast.AstNode)),
    validation_stats: ValidationStats,

    pub const ValidationStats = struct {
        nodes_analyzed: u32,
        errors_found: u32,
        warnings_generated: u32,
        validations_skipped: u32,
        recovery_attempts: u32,

        pub fn init() ValidationStats {
            return ValidationStats{
                .nodes_analyzed = 0,
                .errors_found = 0,
                .warnings_generated = 0,
                .validations_skipped = 0,
                .recovery_attempts = 0,
            };
        }
    };

    pub fn init(allocator: std.mem.Allocator) ValidationCoverage {
        return ValidationCoverage{
            .visited_node_types = std.EnumSet(std.meta.Tag(ast.AstNode)).initEmpty(),
            .missing_implementations = std.ArrayList(std.meta.Tag(ast.AstNode)).init(allocator),
            .validation_stats = ValidationStats.init(),
        };
    }

    pub fn deinit(self: *ValidationCoverage) void {
        self.missing_implementations.deinit();
    }
};

/// Contract analysis context
pub const ContractContext = struct {
    name: []const u8,
    has_init: bool,
    init_is_public: bool,
    storage_variables: std.ArrayList([]const u8),
    functions: std.ArrayList([]const u8),
    events: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, name: []const u8) ContractContext {
        return ContractContext{
            .name = name,
            .has_init = false,
            .init_is_public = false,
            .storage_variables = std.ArrayList([]const u8).init(allocator),
            .functions = std.ArrayList([]const u8).init(allocator),
            .events = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *ContractContext) void {
        self.storage_variables.deinit();
        self.functions.deinit();
        self.events.deinit();
    }
};
