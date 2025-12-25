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
    // for labeled switches with return support
    return_flag_memref: ?c.MlirValue = null,
    return_value_memref: ?c.MlirValue = null,
    // label type for context-aware handling
    label_type: enum { Switch, While, For, Block } = .Block,
};

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
    current_function_return_type: ?c.MlirType,
    current_function_return_type_info: ?lib.ast.Types.TypeInfo,
    ora_dialect: *@import("../dialect.zig").OraDialect,
    label_context: ?*const LabelContext,
    ensures_clauses: []*lib.ast.Expressions.ExprNode = &[_]*lib.ast.Expressions.ExprNode{}, // Ensures clauses to check before returns
    in_try_block: bool = false, // Track if we're inside a try block (returns should not use ora.return)
    try_return_flag_memref: ?c.MlirValue = null, // Memref for return flag in try blocks
    try_return_value_memref: ?c.MlirValue = null, // Memref for return value in try blocks

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, expr_lowerer: *const ExpressionLowerer, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap, locations: LocationTracker, symbol_table: ?*SymbolTable, builtin_registry: ?*const lib.semantics.builtins.BuiltinRegistry, allocator: std.mem.Allocator, function_return_type: ?c.MlirType, function_return_type_info: ?lib.ast.Types.TypeInfo, ora_dialect: *@import("../dialect.zig").OraDialect, ensures_clauses: []*lib.ast.Expressions.ExprNode) StatementLowerer {
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
            .current_function_return_type = function_return_type,
            .current_function_return_type_info = function_return_type_info,
            .ora_dialect = ora_dialect,
            .label_context = null,
            .ensures_clauses = ensures_clauses,
            .in_try_block = false,
        };
    }

    /// Main dispatch function for lowering statements
    pub fn lowerStatement(self: *const StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) LoweringError!void {
        _ = self.fileLoc(self.getStatementSpan(stmt));

        switch (stmt.*) {
            .Return => |ret| {
                try return_stmt.lowerReturn(self, &ret);
            },
            .VariableDecl => |var_decl| {
                std.debug.print("[statement_lowerer] Processing VariableDecl: {s}, region={any}\n", .{ var_decl.name, var_decl.region });
                std.debug.print("[BEFORE MLIR] Variable: {s}\n", .{var_decl.name});
                std.debug.print("  category: {s}\n", .{@tagName(var_decl.type_info.category)});
                std.debug.print("  source: {s}\n", .{@tagName(var_decl.type_info.source)});
                if (var_decl.type_info.ora_type) |_| {
                    std.debug.print("  ora_type: present\n", .{});
                } else {
                    std.debug.print("  ora_type: NULL\n", .{});
                }
                try variables.lowerVariableDecl(self, &var_decl);
            },
            .DestructuringAssignment => |assignment| {
                try assignments.lowerDestructuringAssignment(self, &assignment);
            },
            .CompoundAssignment => |assignment| {
                try assignments.lowerCompoundAssignment(self, &assignment);
            },
            .If => |if_stmt| {
                try control_flow.lowerIf(self, &if_stmt);
            },
            .While => |while_stmt| {
                try control_flow.lowerWhile(self, &while_stmt);
            },
            .ForLoop => |for_stmt| {
                try control_flow.lowerFor(self, &for_stmt);
            },
            .Switch => |switch_stmt| {
                try control_flow.lowerSwitch(self, &switch_stmt);
            },
            .Break => |break_stmt| {
                try labels.lowerBreak(self, &break_stmt);
            },
            .Continue => |continue_stmt| {
                try labels.lowerContinue(self, &continue_stmt);
            },
            .Log => |log_stmt| {
                try primitives.lowerLog(self, &log_stmt);
            },
            .Lock => |lock_stmt| {
                try primitives.lowerLock(self, &lock_stmt);
            },
            .Unlock => |unlock_stmt| {
                try primitives.lowerUnlock(self, &unlock_stmt);
            },
            .Assert => |assert_stmt| {
                try verification.lowerAssert(self, &assert_stmt);
            },
            .TryBlock => |try_stmt| {
                try error_handling.lowerTryBlock(self, &try_stmt);
            },
            .ErrorDecl => |error_decl| {
                try error_handling.lowerErrorDecl(self, &error_decl);
            },
            .Invariant => |invariant| {
                try verification.lowerInvariant(self, &invariant);
            },
            .Requires => |requires| {
                try verification.lowerRequires(self, &requires);
            },
            .Ensures => |ensures| {
                try verification.lowerEnsures(self, &ensures);
            },
            .Assume => |assume| {
                try verification.lowerAssume(self, &assume);
            },
            .Havoc => |havoc| {
                try verification.lowerHavoc(self, &havoc);
            },
            .Expr => |expr| {
                try assignments.lowerExpressionStatement(self, &expr);
            },
            .LabeledBlock => |labeled_block| {
                try labels.lowerLabeledBlock(self, &labeled_block);
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
    pub fn lowerBlockBody(self: *const StatementLowerer, b: lib.ast.Statements.BlockNode, block: c.MlirBlock) LoweringError!bool {
        if (self.symbol_table) |st| {
            st.pushScope() catch {
                std.debug.print("WARNING: Failed to push scope for block\n", .{});
            };
        }

        const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.expr_lowerer.error_handler, self.locations, self.ora_dialect);

        var has_terminator = false;

        for (b.statements) |*s| {
            if (has_terminator) break;

            const is_terminator = switch (s.*) {
                .Break, .Continue, .Return => true,
                else => false,
            };

            var stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, &expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.builtin_registry, self.allocator, self.current_function_return_type, self.current_function_return_type_info, self.ora_dialect, self.ensures_clauses);
            stmt_lowerer.label_context = self.label_context;
            stmt_lowerer.in_try_block = self.in_try_block;
            stmt_lowerer.try_return_flag_memref = self.try_return_flag_memref;
            stmt_lowerer.try_return_value_memref = self.try_return_value_memref;

            stmt_lowerer.lowerStatement(s) catch |err| {
                if (self.symbol_table) |st| {
                    st.popScope();
                }
                return err;
            };

            if (is_terminator) {
                has_terminator = true;
            }
        }

        if (self.symbol_table) |st| {
            st.popScope();
        }

        return has_terminator;
    }

    /// Create file location for operations
    pub fn fileLoc(self: *const StatementLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return self.locations.createLocation(span);
    }
};
