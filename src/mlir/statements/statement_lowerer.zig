// ============================================================================
// Statement Lowering - Core
// ============================================================================
//
// Core StatementLowerer struct and dispatcher.
// Implementation modules are in separate files.
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const ExpressionLowerer = @import("../expressions.zig").ExpressionLowerer;
const ParamMap = @import("../lower.zig").ParamMap;
const StorageMap = @import("../memory.zig").StorageMap;
const LocalVarMap = @import("../lower.zig").LocalVarMap;
const LocationTracker = @import("../lower.zig").LocationTracker;
const MemoryManager = @import("../memory.zig").MemoryManager;
const SymbolTable = @import("../lower.zig").SymbolTable;
const log = @import("log");

// Import implementation modules
const control_flow = @import("control_flow.zig");
const variables = @import("variables.zig");
const assignments = @import("assignments.zig");
const labels = @import("labels.zig");
const error_handling = @import("error_handling.zig");
const primitives = @import("primitives.zig");
const verification = @import("verification.zig");
const return_stmt = @import("return.zig");
const helpers = @import("helpers.zig");

/// Context for labeled control flow (switch, while, for, blocks)
pub const LabelContext = struct {
    label: []const u8,
    // for labeled switches with continue support
    continue_flag_memref: ?c.MlirValue = null,
    value_memref: ?c.MlirValue = null,
    // for labeled blocks (break handling)
    break_flag_memref: ?c.MlirValue = null,
    // for labeled switches with return support
    return_flag_memref: ?c.MlirValue = null,
    return_value_memref: ?c.MlirValue = null,
    // label type for context-aware handling
    label_type: enum { Switch, While, For, Block } = .Block,
    parent: ?*const LabelContext = null,
};

/// Check if we're inside a for/while loop context where we can't use ora.return
fn isInsideLoopContext(label_context: ?*const LabelContext) bool {
    var ctx_opt = label_context;
    while (ctx_opt) |ctx| : (ctx_opt = ctx.parent) {
        switch (ctx.label_type) {
            .For, .While => return true,
            .Block, .Switch => {},
        }
    }
    return false;
}

/// Statement lowering system for converting Ora statements to MLIR operations
pub const StatementLowerer = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    expr_lowerer: *const ExpressionLowerer,
    param_map: ?*const ParamMap,
    storage_map: ?*const StorageMap,
    local_var_map: ?*LocalVarMap,
    locations: LocationTracker,
    symbol_table: ?*SymbolTable,
    builtin_registry: ?*const lib.semantics.builtins.BuiltinRegistry,
    memory_manager: MemoryManager,
    allocator: std.mem.Allocator,
    refinement_guard_cache: ?*std.AutoHashMap(u128, void) = null,
    current_function_return_type: ?c.MlirType,
    current_function_return_type_info: ?lib.ast.Types.TypeInfo,
    ora_dialect: *@import("../dialect.zig").OraDialect,
    function_block: c.MlirBlock,
    label_context: ?*const LabelContext,
    ensures_clauses: []*lib.ast.Expressions.ExprNode = &[_]*lib.ast.Expressions.ExprNode{}, // Ensures clauses to check before returns
    in_try_block: bool = false, // Track if we're inside a try block (returns should not use ora.return)
    try_return_flag_memref: ?c.MlirValue = null, // Memref for return flag in try blocks
    try_return_value_memref: ?c.MlirValue = null, // Memref for return value in try blocks
    active_condition_span: ?lib.ast.SourceSpan = null, // Reuse condition values for assume lowering
    active_condition_value: ?c.MlirValue = null,
    active_condition_safe: bool = false,
    active_condition_expr: ?*const lib.ast.Expressions.ExprNode = null,
    force_stack_memref: bool = false, // Force stack locals to lower as memrefs (e.g. labeled blocks)
    current_func_op: ?c.MlirOperation = null, // Current function operation (for creating new blocks)
    last_block: ?c.MlirBlock = null, // Last block used by lowerBlockBody

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, expr_lowerer: *const ExpressionLowerer, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap, locations: LocationTracker, symbol_table: ?*SymbolTable, builtin_registry: ?*const lib.semantics.builtins.BuiltinRegistry, allocator: std.mem.Allocator, refinement_guard_cache: ?*std.AutoHashMap(u128, void), function_return_type: ?c.MlirType, function_return_type_info: ?lib.ast.Types.TypeInfo, ora_dialect: *@import("../dialect.zig").OraDialect, ensures_clauses: []*lib.ast.Expressions.ExprNode) StatementLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
            .expr_lowerer = expr_lowerer,
            .param_map = param_map,
            .storage_map = storage_map,
            .local_var_map = local_var_map,
            .locations = locations,
            .symbol_table = symbol_table,
            .builtin_registry = builtin_registry,
            .memory_manager = MemoryManager.init(ctx, ora_dialect),
            .allocator = allocator,
            .refinement_guard_cache = refinement_guard_cache,
            .current_function_return_type = function_return_type,
            .current_function_return_type_info = function_return_type_info,
            .ora_dialect = ora_dialect,
            .function_block = block,
            .label_context = null,
            .ensures_clauses = ensures_clauses,
            .in_try_block = false,
            .force_stack_memref = false,
            .active_condition_span = null,
            .active_condition_value = null,
            .active_condition_safe = false,
            .active_condition_expr = null,
        };
    }

    /// Main dispatch function for lowering statements
    /// Returns an optional new block that subsequent statements should use
    pub fn lowerStatement(self: *const StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) LoweringError!?c.MlirBlock {
        _ = self.fileLoc(self.getStatementSpan(stmt));

        switch (stmt.*) {
            .Return => |ret| {
                try return_stmt.lowerReturn(self, &ret);
                return null;
            },
            .VariableDecl => |var_decl| {
                log.debug("[statement_lowerer] Processing VariableDecl: {s}, region={any}\n", .{ var_decl.name, var_decl.region });
                log.debug("[BEFORE MLIR] Variable: {s}\n", .{var_decl.name});
                log.debug("  category: {s}\n", .{@tagName(var_decl.type_info.category)});
                log.debug("  source: {s}\n", .{@tagName(var_decl.type_info.source)});
                if (var_decl.type_info.ora_type) |_| {
                    log.debug("  ora_type: present\n", .{});
                } else {
                    log.debug("  ora_type: NULL\n", .{});
                }
                try variables.lowerVariableDecl(self, &var_decl);
                return null;
            },
            .DestructuringAssignment => |assignment| {
                try assignments.lowerDestructuringAssignment(self, &assignment);
                return null;
            },
            .CompoundAssignment => |assignment| {
                try assignments.lowerCompoundAssignment(self, &assignment);
                return null;
            },
            .If => |if_stmt| {
                return try control_flow.lowerIf(self, &if_stmt);
            },
            .While => |while_stmt| {
                try control_flow.lowerWhile(self, &while_stmt);
                return null;
            },
            .ForLoop => |for_stmt| {
                try control_flow.lowerFor(self, &for_stmt);
                return null;
            },
            .Switch => |switch_stmt| {
                try control_flow.lowerSwitch(self, &switch_stmt);
                return null;
            },
            .Break => |break_stmt| {
                try labels.lowerBreak(self, &break_stmt);
                return null;
            },
            .Continue => |continue_stmt| {
                try labels.lowerContinue(self, &continue_stmt);
                return null;
            },
            .Log => |log_stmt| {
                try primitives.lowerLog(self, &log_stmt);
                return null;
            },
            .Lock => |lock_stmt| {
                try primitives.lowerLock(self, &lock_stmt);
                return null;
            },
            .Unlock => |unlock_stmt| {
                try primitives.lowerUnlock(self, &unlock_stmt);
                return null;
            },
            .Assert => |assert_stmt| {
                try verification.lowerAssert(self, &assert_stmt);
                return null;
            },
            .TryBlock => |try_stmt| {
                try error_handling.lowerTryBlock(self, &try_stmt);
                return null;
            },
            .ErrorDecl => |error_decl| {
                log.err("Error declaration reached statement lowering: {s}\n", .{error_decl.name});
                return LoweringError.UnsupportedStatement;
            },
            .Invariant => |invariant| {
                try verification.lowerInvariant(self, &invariant);
                return null;
            },
            .Requires => |requires| {
                try verification.lowerRequires(self, &requires);
                return null;
            },
            .Ensures => |ensures| {
                try verification.lowerEnsures(self, &ensures);
                return null;
            },
            .Assume => |*assume| {
                try verification.lowerAssume(self, assume);
                return null;
            },
            .Havoc => |havoc| {
                try verification.lowerHavoc(self, &havoc);
                return null;
            },
            .Expr => |expr| {
                try assignments.lowerExpressionStatement(self, &expr);
                return null;
            },
            .LabeledBlock => |labeled_block| {
                try labels.lowerLabeledBlock(self, &labeled_block);
                return null;
            },
        }
    }

    /// Error types for statement lowering
    pub const LoweringError = error{
        UnsupportedStatement,
        TypeMismatch,
        UndefinedSymbol,
        InvalidMemoryRegion,
        MalformedExpression,
        MlirOperationFailed,
        OutOfMemory,
        InvalidControlFlow,
        InvalidLValue,
        InvalidSwitch,
    };

    /// Get the source span for any statement type
    fn getStatementSpan(_: *const StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) lib.ast.SourceSpan {
        return switch (stmt.*) {
            .Return => |ret| ret.span,
            .VariableDecl => |var_decl| var_decl.span,
            .DestructuringAssignment => |assignment| assignment.span,
            .CompoundAssignment => |assignment| assignment.span,
            .If => |if_stmt| if_stmt.span,
            .While => |while_stmt| while_stmt.span,
            .ForLoop => |for_stmt| for_stmt.span,
            .Switch => |switch_stmt| switch_stmt.span,
            .Break => |break_stmt| break_stmt.span,
            .Continue => |continue_stmt| continue_stmt.span,
            .Log => |log_stmt| log_stmt.span,
            .Lock => |lock_stmt| lock_stmt.span,
            .Unlock => |unlock_stmt| unlock_stmt.span,
            .Assert => |assert_stmt| assert_stmt.span,
            .TryBlock => |try_stmt| try_stmt.span,
            .ErrorDecl => |error_decl| error_decl.span,
            .Invariant => |invariant| invariant.span,
            .Requires => |requires| requires.span,
            .Ensures => |ensures| ensures.span,
            .Assume => |assume| assume.span,
            .Havoc => |havoc| havoc.span,
            .Expr => |expr| getExpressionSpan(&expr),
            .LabeledBlock => |labeled_block| labeled_block.span,
        };
    }

    /// Get the source span for any expression type
    fn getExpressionSpan(_: *const lib.ast.Statements.ExprNode) lib.ast.SourceSpan {
        return lib.ast.SourceSpan{ .line = 1, .column = 1, .length = 0 };
    }

    /// Lower block body with proper error handling and location tracking
    pub fn lowerBlockBody(self: *const StatementLowerer, b: lib.ast.Statements.BlockNode, initial_block: c.MlirBlock) LoweringError!bool {
        if (self.symbol_table) |st| {
            st.pushScope() catch {
                log.debug("WARNING: Failed to push scope for block\n", .{});
            };
        }

        // Track current block - may change if partial returns create continue blocks
        var current_block = initial_block;

        var expr_lowerer = ExpressionLowerer.init(self.ctx, current_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.expr_lowerer.error_handler, self.locations, self.ora_dialect);
        expr_lowerer.refinement_base_cache = self.expr_lowerer.refinement_base_cache;
        expr_lowerer.refinement_guard_cache = self.refinement_guard_cache;
        expr_lowerer.current_function_return_type = self.current_function_return_type;
        expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
        expr_lowerer.in_try_block = self.in_try_block;

        var has_terminator = false;

        var stmt_idx: usize = 0;
        while (stmt_idx < b.statements.len) : (stmt_idx += 1) {
            if (has_terminator) break;

            const s = &b.statements[stmt_idx];
            const is_terminator = switch (s.*) {
                .Break, .Continue, .Return => true,
                else => false,
            };

            // Check for pattern: if (with one branch returning) followed by return
            // Transform: if (c) { return x; } return y; -> if (c) { return x; } else { return y; }
            if (s.* == .If and stmt_idx + 1 < b.statements.len) {
                const if_stmt = &s.If;
                const next_stmt = &b.statements[stmt_idx + 1];
                const then_returns = helpers.blockHasReturn(self, if_stmt.then_branch);
                const else_returns = if (if_stmt.else_branch) |else_branch| helpers.blockHasReturn(self, else_branch) else false;

                // If then returns but no else, and next statement is return, merge them
                if (then_returns and !else_returns and if_stmt.else_branch == null and next_stmt.* == .Return) {
                    expr_lowerer.block = current_block;
                    var stmt_lowerer = StatementLowerer.init(self.ctx, current_block, self.type_mapper, &expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.builtin_registry, self.allocator, self.refinement_guard_cache, self.current_function_return_type, self.current_function_return_type_info, self.ora_dialect, self.ensures_clauses);
                    stmt_lowerer.force_stack_memref = self.force_stack_memref;
                    stmt_lowerer.label_context = self.label_context;
                    stmt_lowerer.in_try_block = self.in_try_block;
                    stmt_lowerer.current_func_op = self.current_func_op;

                    // Use lowerIfWithFollowingReturn which handles this pattern
                    try control_flow.lowerIfWithFollowingReturn(&stmt_lowerer, if_stmt, &next_stmt.Return);
                    has_terminator = true;
                    stmt_idx += 1; // Skip the return statement as we've already handled it
                    continue;
                }
            }

            // Update expr_lowerer's block if it changed
            expr_lowerer.block = current_block;

            var stmt_lowerer = StatementLowerer.init(self.ctx, current_block, self.type_mapper, &expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.builtin_registry, self.allocator, self.refinement_guard_cache, self.current_function_return_type, self.current_function_return_type_info, self.ora_dialect, self.ensures_clauses);
            stmt_lowerer.force_stack_memref = self.force_stack_memref;
            stmt_lowerer.label_context = self.label_context;
            stmt_lowerer.in_try_block = self.in_try_block;
            stmt_lowerer.try_return_flag_memref = self.try_return_flag_memref;
            stmt_lowerer.try_return_value_memref = self.try_return_value_memref;
            stmt_lowerer.current_func_op = self.current_func_op;

            const maybe_new_block = stmt_lowerer.lowerStatement(s) catch |err| {
                if (self.symbol_table) |st| {
                    st.popScope();
                }
                return err;
            };

            // If statement returns a new block, subsequent statements go there
            const created_new_block = maybe_new_block != null;
            if (maybe_new_block) |new_block| {
                log.debug("[lowerBlockBody] Got new block from lowerStatement, updating current_block\n", .{});
                current_block = new_block;
                if (stmt_idx == b.statements.len - 1 and !helpers.blockEndsWithTerminator(&stmt_lowerer, current_block)) {
                    const next_block = c.oraBlockGetNextInRegion(current_block);
                    if (c.mlirBlockIsNull(next_block)) {
                        log.debug("[lowerBlockBody] No next block in region for tail continue block\n", .{});
                    } else {
                        const loc = self.fileLoc(self.getStatementSpan(s));
                        const br = self.ora_dialect.createCfBr(next_block, loc);
                        h.appendOp(current_block, br);
                        has_terminator = true;
                    }
                }
            }

            if (is_terminator or (!created_new_block and helpers.blockEndsWithTerminator(&stmt_lowerer, current_block))) {
                has_terminator = true;
            } else if (s.* == .TryBlock and stmt_idx == b.statements.len - 1 and self.current_function_return_type != null and !isInsideLoopContext(self.label_context)) {
                // Only generate return after try block if NOT inside a for/while loop
                // (scf.for/scf.while require scf.yield, not ora.return)
                const loc = self.fileLoc(self.getStatementSpan(s));
                const ret_type = self.current_function_return_type.?;
                const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                const ret_op = self.ora_dialect.createFuncReturnWithValue(default_val, loc);
                h.appendOp(current_block, ret_op);
                has_terminator = true;
            }
        }

        if (!has_terminator and b.statements.len > 0) {
            const last_stmt = &b.statements[b.statements.len - 1];
            if (last_stmt.* == .Return) {
                expr_lowerer.block = current_block;
                var stmt_lowerer = StatementLowerer.init(self.ctx, current_block, self.type_mapper, &expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.builtin_registry, self.allocator, self.refinement_guard_cache, self.current_function_return_type, self.current_function_return_type_info, self.ora_dialect, self.ensures_clauses);
                stmt_lowerer.force_stack_memref = self.force_stack_memref;
                stmt_lowerer.label_context = self.label_context;
                stmt_lowerer.in_try_block = self.in_try_block;
                stmt_lowerer.try_return_flag_memref = self.try_return_flag_memref;
                stmt_lowerer.try_return_value_memref = self.try_return_value_memref;
                stmt_lowerer.current_func_op = self.current_func_op;
                try return_stmt.lowerReturn(&stmt_lowerer, &last_stmt.Return);
                has_terminator = true;
            }
        }

        if (self.symbol_table) |st| {
            st.popScope();
        }

        @constCast(self).last_block = current_block;
        return has_terminator;
    }

    fn fixEmptyBlocksInOp(self: *const StatementLowerer, op: c.MlirOperation) void {
        const num_regions = c.oraOperationGetNumRegions(op);
        var region_idx: usize = 0;
        while (region_idx < num_regions) : (region_idx += 1) {
            const region = c.oraOperationGetRegion(op, region_idx);
            fixEmptyBlocksInRegion(self, region);
        }
    }

    fn fixEmptyBlocksInRegion(self: *const StatementLowerer, region: c.MlirRegion) void {
        if (c.oraRegionIsNull(region)) return;

        var block = c.oraRegionGetFirstBlock(region);
        while (!c.mlirBlockIsNull(block)) : (block = c.oraBlockGetNextInRegion(block)) {
            var op = c.oraBlockGetFirstOperation(block);
            while (!c.oraOperationIsNull(op)) : (op = c.oraOperationGetNextInBlock(op)) {
                fixEmptyBlocksInOp(self, op);
            }

            if (!helpers.blockEndsWithTerminator(self, block)) {
                const first_op = c.oraBlockGetFirstOperation(block);
                if (c.oraOperationIsNull(first_op)) {
                    const next_block = c.oraBlockGetNextInRegion(block);
                    if (!c.mlirBlockIsNull(next_block)) {
                        // Only insert cf.br in function CFG regions (not in region-based ops).
                        const parent_op = c.mlirBlockGetParentOperation(block);
                        if (!c.oraOperationIsNull(parent_op)) {
                            const name_ref = c.oraOperationGetName(parent_op);
                            if (name_ref.data != null and std.mem.eql(u8, name_ref.data[0..name_ref.length], "func.func")) {
                                const br = self.ora_dialect.createCfBr(next_block, h.unknownLoc(self.ctx));
                                h.appendOp(block, br);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn fixEmptyBlocks(self: *const StatementLowerer) void {
        if (self.current_func_op) |func_op| {
            if (!c.oraOperationIsNull(func_op)) {
                fixEmptyBlocksInOp(self, func_op);
            }
        }
    }

    /// Create file location for operations
    pub fn fileLoc(self: *const StatementLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return self.locations.createLocation(span);
    }
};
