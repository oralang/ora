// Re-export all public interfaces from modular semantics system
pub const semantics_core = @import("semantics/semantics_core.zig");
pub const semantics_errors = @import("semantics/semantics_errors.zig");
pub const semantics_state = @import("semantics/semantics_state.zig");
pub const semantics_memory_safety = @import("semantics/semantics_memory_safety.zig");
pub const semantics_contract_analyzer = @import("semantics/semantics_contract_analyzer.zig");
pub const semantics_function_analyzer = @import("semantics/semantics_function_analyzer.zig");
pub const semantics_expression_analyzer = @import("semantics/semantics_expression_analyzer.zig");
pub const semantics_import_analyzer = @import("semantics/semantics_import_analyzer.zig");
pub const semantics_type_integration = @import("semantics/semantics_type_integration.zig");
pub const semantics_struct_validator = @import("semantics/semantics_struct_validator.zig");
pub const semantics_memory_region_validator = @import("semantics/semantics_memory_region_validator.zig");
pub const semantics_immutable_tracker = @import("semantics/semantics_immutable_tracker.zig");
pub const semantics_circular_dependency_detector = @import("semantics/semantics_circular_dependency_detector.zig");
pub const semantics_builtin_functions = @import("semantics/semantics_builtin_functions.zig");
pub const semantics_diagnostics = @import("semantics/semantics_diagnostics.zig");
pub const semantics_coverage = @import("semantics/semantics_coverage.zig");
pub const semantics_recovery = @import("semantics/semantics_recovery.zig");
pub const semantics_performance = @import("semantics/semantics_performance.zig");
pub const semantics_utils = @import("semantics/semantics_utils.zig");

// Re-export main types for backward compatibility
pub const SemanticAnalyzer = semantics_core.SemanticAnalyzer;
pub const SemanticError = semantics_errors.SemanticError;
pub const Diagnostic = semantics_errors.Diagnostic;
pub const DiagnosticContext = semantics_errors.DiagnosticContext;
pub const AnalysisState = semantics_state.AnalysisState;
pub const ValidationCoverage = semantics_state.ValidationCoverage;
pub const ContractContext = semantics_state.ContractContext;
