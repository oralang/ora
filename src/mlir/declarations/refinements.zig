// ============================================================================
// Declaration Lowering - Refinement Guards
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const StatementLowerer = @import("../statements.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const stmt_helpers = @import("../statements/helpers.zig");
const ExpressionLowerer = @import("../expressions/mod.zig").ExpressionLowerer;

/// Insert refinement guard for function parameters and other declaration-level values.
/// Routed through the shared statement helper to keep guard generation consistent.
pub fn insertRefinementGuard(
    self: *const DeclarationLowerer,
    block: c.MlirBlock,
    value: c.MlirValue,
    ora_type: lib.ast.Types.OraType,
    span: lib.ast.SourceSpan,
    var_name: ?[]const u8,
    refinement_base_cache: ?*std.AutoHashMap(usize, c.MlirValue),
    refinement_guard_cache: ?*std.AutoHashMap(u128, void),
) LoweringError!void {
    var expr_lowerer = ExpressionLowerer.init(
        self.ctx,
        block,
        self.type_mapper,
        null,
        null,
        null,
        self.symbol_table,
        self.builtin_registry,
        self.error_handler,
        self.locations,
        self.ora_dialect,
    );
    expr_lowerer.refinement_base_cache = refinement_base_cache;
    expr_lowerer.refinement_guard_cache = refinement_guard_cache;
    var stmt_lowerer = StatementLowerer.init(
        self.ctx,
        block,
        self.type_mapper,
        &expr_lowerer,
        null,
        null,
        null,
        self.locations,
        self.symbol_table,
        self.builtin_registry,
        std.heap.page_allocator,
        refinement_guard_cache,
        null,
        null,
        self.ora_dialect,
        &[_]*lib.ast.Expressions.ExprNode{},
    );
    _ = try stmt_helpers.insertRefinementGuard(&stmt_lowerer, value, ora_type, span, var_name, false);
}
