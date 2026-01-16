// ============================================================================
// Type Resolver - Main Orchestrator
// ============================================================================
// Phase 5: Pure coordination - delegates to sub-modules
// ============================================================================

const std = @import("std");
// Import AST types - no circular dependency since ast.zig no longer imports us
const ast = @import("../../ast.zig");
const SourceSpan = @import("../../ast/source_span.zig").SourceSpan;
const Statements = @import("../statements.zig");
const TypeInfo = @import("../type_info.zig").TypeInfo;
const OraType = @import("../type_info.zig").OraType;
const state = @import("../../semantics/state.zig");
const SymbolTable = state.SymbolTable;
const Scope = state.Scope;
const semantics = @import("../../semantics.zig");
const FunctionEffect = semantics.state.FunctionEffect;
const mergeEffects = core.mergeEffects;

// Type aliases for clarity (use direct references to avoid comptime issues)
const AstNode = ast.AstNode;
const FunctionNode = ast.FunctionNode;
const ContractNode = ast.ContractNode;
const EnumDeclNode = ast.EnumDeclNode;
const ConstantNode = ast.ConstantNode;

// Sub-modules
const core = @import("core/mod.zig");
const refinements = @import("refinements/mod.zig");
const validation = @import("validation/mod.zig");
const utils = @import("utils/mod.zig");
const log = @import("log");

// Re-export core types
pub const Typed = core.Typed;
pub const Effect = core.Effect;
pub const LockDelta = core.LockDelta;
pub const Obligation = core.Obligation;
pub const TypeContext = core.TypeContext;

const SpecContext = enum { None, Requires, Ensures, Invariant };

fn validateSpecUsageExpr(expr_node: *ast.Expressions.ExprNode, ctx: SpecContext) ?SourceSpan {
    switch (expr_node.*) {
        .Quantified => if (ctx == .None) return expr_node.Quantified.span,
        .Old => if (ctx != .Ensures) return expr_node.Old.span,
        else => {},
    }

    switch (expr_node.*) {
        .Binary => |*b| {
            if (validateSpecUsageExpr(b.lhs, ctx)) |sp| return sp;
            if (validateSpecUsageExpr(b.rhs, ctx)) |sp| return sp;
        },
        .Unary => |*u| {
            if (validateSpecUsageExpr(u.operand, ctx)) |sp| return sp;
        },
        .Assignment => |*a| {
            if (validateSpecUsageExpr(a.value, ctx)) |sp| return sp;
        },
        .Call => |*c| {
            if (validateSpecUsageExpr(c.callee, ctx)) |sp| return sp;
            for (c.arguments) |arg| if (validateSpecUsageExpr(arg, ctx)) |sp| return sp;
        },
        .Tuple => |*t| {
            for (t.elements) |e| if (validateSpecUsageExpr(e, ctx)) |sp| return sp;
        },
        else => {},
    }

    return null;
}

/// Type resolution errors
pub const TypeResolutionError = error{
    UnknownType,
    TypeMismatch,
    RegionMismatch,
    CircularReference,
    InvalidEnumValue,
    InvalidSpecUsage,
    InvalidErrorUsage,
    OutOfMemory,
    IncompatibleTypes,
    UndefinedIdentifier,
    UnresolvedType,
    ErrorUnionOutsideTry,
};

/// Main type resolver orchestrator
pub const TypeResolver = struct {
    allocator: std.mem.Allocator,
    type_storage_allocator: std.mem.Allocator,
    symbol_table: *SymbolTable,
    current_scope: ?*Scope = null,

    // sub-systems
    core_resolver: core.CoreResolver,
    refinement_system: refinements.RefinementSystem,
    validation_system: validation.ValidationSystem,
    utils_system: utils.Utils,

    // function registry for call resolution
    function_registry: std.StringHashMap(*FunctionNode),

    pub fn init(allocator: std.mem.Allocator, type_storage_allocator: std.mem.Allocator, symbol_table: *SymbolTable) TypeResolver {
        var utils_sys = utils.Utils.init(allocator);
        var refinement_sys = refinements.RefinementSystem.init(allocator, &utils_sys);
        var validation_sys = validation.ValidationSystem.init(&refinement_sys, &utils_sys);
        const core_res = core.CoreResolver.init(allocator, type_storage_allocator, symbol_table, &validation_sys, &utils_sys, &refinement_sys);

        return TypeResolver{
            .allocator = allocator,
            .type_storage_allocator = type_storage_allocator,
            .symbol_table = symbol_table,
            .current_scope = symbol_table.root,
            .core_resolver = core_res,
            .refinement_system = refinement_sys,
            .validation_system = validation_sys,
            .utils_system = utils_sys,
            .function_registry = std.StringHashMap(*FunctionNode).init(allocator),
        };
    }

    pub fn deinit(self: *TypeResolver) void {
        self.function_registry.deinit();
        self.core_resolver.deinit();
        self.refinement_system.deinit();
        self.validation_system.deinit();
        self.utils_system.deinit();
    }

    /// Resolve types for an entire AST (public API)
    pub fn resolveTypes(self: *TypeResolver, nodes: []AstNode) TypeResolutionError!void {
        // first pass: Build function registry for argument validation
        for (nodes) |*node| {
            try self.registerFunctions(node);
        }

        // second pass: Resolve types
        for (nodes) |*node| {
            try self.resolveNodeTypes(node, core.TypeContext{});
        }
    }

    /// Resolve types for a single node (public API)
    pub fn resolveNodeTypes(self: *TypeResolver, node: *AstNode, context: core.TypeContext) TypeResolutionError!void {
        // update core resolver's current scope and function registry
        self.core_resolver.current_scope = self.current_scope;
        // type-erase function registry to avoid circular dependency
        self.core_resolver.function_registry = @ptrCast(&self.function_registry);

        switch (node.*) {
            .EnumDecl => |*enum_decl| {
                try self.resolveEnumDecl(enum_decl, context);
            },
            .Contract => |*contract| {
                try self.resolveContract(contract, context);
            },
            .Function => |*function| {
                try self.resolveFunction(function, context);
            },
            .LogDecl => |*log_decl| {
                try self.resolveLogDecl(log_decl, context);
            },
            .VariableDecl => |*var_decl| {
                // convert to StmtNode for statement resolver
                var stmt_node = Statements.StmtNode{ .VariableDecl = var_decl.* };
                var typed = try self.core_resolver.resolveStatement(&stmt_node, context);
                defer typed.deinit(self.allocator);
            },
            .Constant => |*constant| {
                try self.resolveConstant(constant, context);
            },
            .ErrorDecl => |*error_decl| {
                try self.resolveErrorDecl(error_decl, context);
            },
            else => {
                // other node types handled elsewhere
            },
        }
    }

    // ============================================================================
    // node Type Resolvers
    // ============================================================================

    fn resolveEnumDecl(
        self: *TypeResolver,
        enum_decl: *EnumDeclNode,
        context: core.TypeContext,
    ) TypeResolutionError!void {
        _ = self;
        _ = context;
        for (enum_decl.variants) |*variant| {
            if (variant.value) |*value_expr| {
                _ = value_expr;
                // note: We intentionally skip type-resolving enum value
                // expressions here. Their semantics (including references
                // between variants) are handled by the enum system itself, and
                // attempting to resolve identifiers like `Basic` or `Advanced`
                // via the regular symbol table leads to spurious
                // undefinedIdentifier errors during parsing.
            }
        }
    }

    fn resolveConstant(
        self: *TypeResolver,
        constant: *ast.ConstantNode,
        _: core.TypeContext,
    ) TypeResolutionError!void {
        // resolve the constant's value expression
        var value_typed = try self.core_resolver.synthExpr(constant.value);
        defer value_typed.deinit(self.allocator);

        // if type is not explicit, infer from value
        if (!constant.typ.isResolved()) {
            constant.typ = value_typed.ty;
        } else {
            // type is explicit, check value against it
            var checked = try self.core_resolver.checkExpr(constant.value, constant.typ);
            defer checked.deinit(self.allocator);
        }
        try self.core_resolver.validateErrorUnionType(constant.typ);

        // update symbol table with resolved type
        const scope = if (self.current_scope) |s| s else self.symbol_table.root;
        _ = self.symbol_table.updateSymbolType(scope, constant.name, constant.typ, false) catch {};
    }

    fn resolveErrorDecl(
        self: *TypeResolver,
        error_decl: *Statements.ErrorDeclNode,
        _: core.TypeContext,
    ) TypeResolutionError!void {
        if (error_decl.parameters) |params| {
            for (params) |*param| {
                if (param.type_info.ora_type) |ot| {
                    if (ot == .struct_type or ot == .enum_type or ot == .contract_type) {
                        const type_name = switch (ot) {
                            .struct_type => ot.struct_type,
                            .enum_type => ot.enum_type,
                            .contract_type => ot.contract_type,
                            else => unreachable,
                        };
                        const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
                        const type_symbol = SymbolTable.findUp(root_scope, type_name);
                        if (type_symbol) |tsym| {
                            switch (tsym.kind) {
                                .Enum => {
                                    param.type_info.ora_type = OraType{ .enum_type = type_name };
                                    param.type_info.category = .Enum;
                                },
                                .Struct => {
                                    param.type_info.ora_type = OraType{ .struct_type = type_name };
                                    param.type_info.category = .Struct;
                                },
                                .Contract => {
                                    param.type_info.ora_type = OraType{ .contract_type = type_name };
                                    param.type_info.category = .Contract;
                                },
                                else => return TypeResolutionError.UnknownType,
                            }
                        } else {
                            return TypeResolutionError.UnknownType;
                        }
                    } else {
                        param.type_info.category = ot.getCategory();
                    }
                }
                if (!param.type_info.isResolved()) {
                    return TypeResolutionError.UnresolvedType;
                }
            }
        }
    }

    fn resolveLogDecl(
        self: *TypeResolver,
        log_decl: *ast.LogDeclNode,
        _: core.TypeContext,
    ) TypeResolutionError!void {
        for (log_decl.fields) |*field| {
            if (field.type_info.ora_type) |ot| {
                if (ot == .struct_type) {
                    const type_name = ot.struct_type;
                    const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
                    const type_symbol = SymbolTable.findUp(root_scope, type_name);
                    if (type_symbol) |tsym| {
                        switch (tsym.kind) {
                            .Enum => {
                                field.type_info.ora_type = OraType{ .enum_type = type_name };
                                field.type_info.category = .Enum;
                            },
                            .Struct => {
                                field.type_info.ora_type = OraType{ .struct_type = type_name };
                                field.type_info.category = .Struct;
                            },
                            .Contract => {
                                field.type_info.ora_type = OraType{ .contract_type = type_name };
                                field.type_info.category = .Contract;
                            },
                            else => return TypeResolutionError.UnknownType,
                        }
                    } else {
                        return TypeResolutionError.UnknownType;
                    }
                } else {
                    var derived_category = ot.getCategory();
                    if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                        derived_category = .ErrorUnion;
                    }
                    field.type_info.category = derived_category;
                }
            } else {
                return TypeResolutionError.UnresolvedType;
            }
            if (!field.type_info.isResolved()) {
                return TypeResolutionError.UnresolvedType;
            }
        }
    }

    fn resolveContract(
        self: *TypeResolver,
        contract: *ContractNode,
        context: core.TypeContext,
    ) TypeResolutionError!void {
        // set current scope to contract scope if it exists
        const prev_scope = self.current_scope;
        const prev_contract_name = self.core_resolver.current_contract_name;
        if (self.symbol_table.contract_scopes.get(contract.name)) |contract_scope| {
            self.current_scope = contract_scope;
            self.core_resolver.current_scope = contract_scope;
            log.debug("[resolveContract] Set current_scope to contract scope '{s}'\n", .{contract.name});
        } else {
            log.debug("[resolveContract] WARNING: Contract scope '{s}' not found!\n", .{contract.name});
        }
        self.core_resolver.current_contract_name = contract.name;
        const prev_return_type = self.core_resolver.current_function_return_type;
        defer {
            self.current_scope = prev_scope;
            self.core_resolver.current_scope = prev_scope;
            self.core_resolver.current_function_return_type = prev_return_type;
            self.core_resolver.current_contract_name = prev_contract_name;
        }

        // resolve constants first, then other members
        // this ensures constants are available when resolving functions that use them
        if (self.symbol_table.contract_log_signatures.getPtr(contract.name) == null) {
            const log_map = std.StringHashMap([]const ast.LogField).init(self.allocator);
            try self.symbol_table.contract_log_signatures.put(contract.name, log_map);
        }
        for (contract.body) |*child| {
            if (child.* == .LogDecl) {
                const log_decl = &child.LogDecl;
                try self.resolveLogDecl(log_decl, context);
                if (self.symbol_table.contract_log_signatures.getPtr(contract.name)) |log_map| {
                    try log_map.put(log_decl.name, log_decl.fields);
                }
            }
        }
        for (contract.body) |*child| {
            if (child.* == .Constant) {
                try self.resolveNodeTypes(child, context);
            }
        }
        // then resolve everything else
        for (contract.body) |*child| {
            if (child.* != .Constant) {
                try self.resolveNodeTypes(child, context);
            }
        }
    }

    fn resolveFunction(
        self: *TypeResolver,
        function: *FunctionNode,
        context: core.TypeContext,
    ) TypeResolutionError!void {
        // parameters should already have explicit types, just validate them
        for (function.parameters) |*param| {
            if (param.type_info.ora_type) |ot| {
                if (ot == .struct_type) {
                    const type_name = ot.struct_type;
                    const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
                    const type_symbol = SymbolTable.findUp(root_scope, type_name);
                    if (type_symbol) |tsym| {
                        if (tsym.kind == .Enum) {
                            param.type_info.ora_type = OraType{ .enum_type = type_name };
                            param.type_info.category = .Enum;
                        } else {
                            var derived_category = ot.getCategory();
                            if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                                derived_category = .ErrorUnion;
                            }
                            param.type_info.category = derived_category;
                        }
                    } else {
                        var derived_category = ot.getCategory();
                        if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                            derived_category = .ErrorUnion;
                        }
                        param.type_info.category = derived_category;
                    }
                } else {
                    var derived_category = ot.getCategory();
                    if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                        derived_category = .ErrorUnion;
                    }
                    param.type_info.category = derived_category;
                }
            }
            if (!param.type_info.isResolved()) {
                return TypeResolutionError.UnresolvedType;
            }
            try self.core_resolver.validateErrorUnionType(param.type_info);
        }

        // return type should be explicit or void
        if (function.return_type_info) |*ret_type| {
            // fix custom type names: if parser assumed struct_type but it's actually enum_type
            if (ret_type.ora_type) |ot| {
                if (ot == .struct_type) {
                    const type_name = ot.struct_type;
                    // look up symbol to see if it's actually an enum
                    const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
                    const type_symbol = SymbolTable.findUp(root_scope, type_name);
                    if (type_symbol) |tsym| {
                        if (tsym.kind == .Enum) {
                            // fix: change struct_type to enum_type
                            ret_type.ora_type = OraType{ .enum_type = type_name };
                            ret_type.category = .Enum;
                        } else {
                            // use the category from the ora_type
                            var derived_category = ot.getCategory();
                            if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                                derived_category = .ErrorUnion;
                            }
                            ret_type.category = derived_category;
                        }
                    } else {
                        // type not found, use category from ora_type
                        var derived_category = ot.getCategory();
                        if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                            derived_category = .ErrorUnion;
                        }
                        ret_type.category = derived_category;
                    }
                } else {
                    // use the category from the ora_type
                    var derived_category = ot.getCategory();
                    if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                        derived_category = .ErrorUnion;
                    }
                    ret_type.category = derived_category;
                }
            }
            if (!ret_type.isResolved()) {
                return TypeResolutionError.UnresolvedType;
            }
            try self.core_resolver.validateErrorUnionType(ret_type.*);
        }

        // get or create function scope for variable declarations
        const prev_scope = self.current_scope;
        var func_scope: ?*Scope = null;
        if (self.symbol_table.function_scopes.get(function.name)) |scope| {
            func_scope = scope;
            self.current_scope = scope;
            self.core_resolver.current_scope = scope;
            if (scope.parent == null and prev_scope != null) {
                scope.parent = prev_scope;
            }
        } else {
            // create function scope if it doesn't exist
            // use current_scope as parent (should be contract scope if inside contract)
            const parent_scope = self.current_scope;
            const new_scope = try self.allocator.create(Scope);
            new_scope.* = Scope.init(self.allocator, parent_scope, function.name);
            try self.symbol_table.scopes.append(self.allocator, new_scope);
            try self.symbol_table.function_scopes.put(function.name, new_scope);
            func_scope = new_scope;
            self.current_scope = new_scope;
            self.core_resolver.current_scope = new_scope;
        }
        // ensure parameters exist in function scope (and are marked as calldata)
        if (func_scope) |scope| {
            for (function.parameters) |*param| {
                param.type_info.region = .Calldata;
                if (scope.findInCurrent(param.name)) |idx| {
                    scope.symbols.items[idx].typ = param.type_info;
                    scope.symbols.items[idx].region = .Calldata;
                    scope.symbols.items[idx].mutable = param.is_mutable;
                } else {
                    const param_symbol = semantics.state.Symbol{
                        .name = param.name,
                        .kind = .Param,
                        .typ = param.type_info,
                        .span = param.span,
                        .mutable = param.is_mutable,
                        .region = .Calldata,
                    };
                    _ = try self.symbol_table.declare(scope, param_symbol);
                }
            }
        }
        defer {
            self.current_scope = prev_scope;
            self.core_resolver.current_scope = prev_scope;
        }

        // create context for function body with return type
        var func_context = context;
        func_context.function_return_type = function.return_type_info;

        // resolve requires/ensures expressions
        for (function.requires_clauses) |clause| {
            if (validateSpecUsageExpr(clause, .Requires)) |_| return TypeResolutionError.InvalidSpecUsage;
            var typed = try self.core_resolver.synthExpr(clause);
            defer typed.deinit(self.allocator);
        }
        for (function.ensures_clauses) |clause| {
            if (validateSpecUsageExpr(clause, .Ensures)) |_| return TypeResolutionError.InvalidSpecUsage;
            var typed = try self.core_resolver.synthExpr(clause);
            defer typed.deinit(self.allocator);
        }

        // resolve all statements in function body
        self.core_resolver.current_function_return_type = function.return_type_info;
        // note: We don't set block scopes here to avoid double-frees during deinit
        // variables declared in blocks will be found via findUp from the function scope
        var func_effect = Effect.pure();
        for (function.body.statements) |*stmt| {
            var typed = try self.core_resolver.resolveStatement(stmt, func_context);
            defer typed.deinit(self.allocator);
            const eff = core.takeEffect(&typed);
            mergeEffects(self.allocator, &func_effect, eff);
        }
        // note: Function call argument validation happens in synthCall
        // via function_registry which is set on core_resolver

        const stored_effect = functionEffectFromCore(self.allocator, func_effect);
        if (self.symbol_table.function_effects.getPtr(function.name)) |existing| {
            existing.deinit(self.allocator);
            existing.* = stored_effect;
        } else {
            try self.symbol_table.function_effects.put(function.name, stored_effect);
        }
        func_effect.deinit(self.allocator);

        // validate return statements in function body
        try self.validateReturnStatements(function);
    }


    /// Validate all return statements in a function
    fn validateReturnStatements(
        self: *TypeResolver,
        function: *FunctionNode,
    ) TypeResolutionError!void {
        var expected_return_type = function.return_type_info;

        if (expected_return_type == null) {
            var inferred: ?TypeInfo = null;
            var saw_value = false;
            var saw_void = false;
            try self.inferReturnTypeInBlock(&function.body, &inferred, &saw_value, &saw_void);

            if (saw_value and saw_void) {
                return TypeResolutionError.TypeMismatch;
            }
            if (saw_value) {
                expected_return_type = inferred;
                function.return_type_info = inferred;
            } else if (saw_void) {
                const void_ty = TypeInfo{
                    .category = .Void,
                    .ora_type = OraType.void,
                    .source = .inferred,
                    .span = function.span,
                };
                expected_return_type = void_ty;
                function.return_type_info = void_ty;
            }
        }

        if (expected_return_type) |ret_ty| {
            log.debug("[validateReturnStatements] Function '{s}' return type: category={s}\n", .{ function.name, @tagName(ret_ty.category) });
        } else {
            log.debug("[validateReturnStatements] Function '{s}' has no return type\n", .{function.name});
        }

        // walk through all statements in the function body
        for (function.body.statements) |*stmt| {
            try self.validateReturnInStatement(stmt, expected_return_type);
        }
    }

    fn inferReturnTypeInStatement(
        self: *TypeResolver,
        stmt: *Statements.StmtNode,
        inferred: *?TypeInfo,
        saw_value: *bool,
        saw_void: *bool,
    ) TypeResolutionError!void {
        switch (stmt.*) {
            .Return => |*ret| {
                if (ret.value) |*value_expr| {
                    var typed = try self.core_resolver.synthExpr(value_expr);
                    defer typed.deinit(self.allocator);

                    if (!typed.ty.isResolved()) {
                        return TypeResolutionError.UnresolvedType;
                    }

                    saw_value.* = true;
                    if (inferred.*) |current| {
                        if (!self.validation_system.isAssignable(typed.ty, current)) {
                            return TypeResolutionError.TypeMismatch;
                        }
                    } else {
                        inferred.* = typed.ty;
                    }
                } else {
                    saw_void.* = true;
                }
            },
            .If => |*if_stmt| {
                try self.inferReturnTypeInBlock(&if_stmt.then_branch, inferred, saw_value, saw_void);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.inferReturnTypeInBlock(else_branch, inferred, saw_value, saw_void);
                }
            },
            .While => |*while_stmt| {
                try self.inferReturnTypeInBlock(&while_stmt.body, inferred, saw_value, saw_void);
            },
            .ForLoop => |*for_stmt| {
                try self.inferReturnTypeInBlock(&for_stmt.body, inferred, saw_value, saw_void);
            },
            .LabeledBlock => |*labeled_block| {
                try self.inferReturnTypeInBlock(&labeled_block.block, inferred, saw_value, saw_void);
            },
            .TryBlock => |*try_block| {
                try self.inferReturnTypeInBlock(&try_block.try_block, inferred, saw_value, saw_void);
                if (try_block.catch_block) |*catch_block| {
                    try self.inferReturnTypeInBlock(&catch_block.block, inferred, saw_value, saw_void);
                }
            },
            else => {},
        }
    }

    fn inferReturnTypeInBlock(
        self: *TypeResolver,
        block: *Statements.BlockNode,
        inferred: *?TypeInfo,
        saw_value: *bool,
        saw_void: *bool,
    ) TypeResolutionError!void {
        for (block.statements) |*stmt| {
            try self.inferReturnTypeInStatement(stmt, inferred, saw_value, saw_void);
        }
    }

    /// Validate return statement within a statement
    fn validateReturnInStatement(
        self: *TypeResolver,
        stmt: *Statements.StmtNode,
        expected_return_type: ?TypeInfo,
    ) TypeResolutionError!void {
        switch (stmt.*) {
            .Return => |*ret| {
                if (ret.value) |*value_expr| {
                    // check expression against expected return type
                    if (expected_return_type) |return_type| {
                        var checked = try self.core_resolver.checkExpr(value_expr, return_type);
                        defer checked.deinit(self.allocator);
                    } else {
                        // no return type expected - synthesize
                        var typed = try self.core_resolver.synthExpr(value_expr);
                        defer typed.deinit(self.allocator);
                    }
                } else {
                    // void return - check if function expects void
                    if (expected_return_type != null and expected_return_type.?.category != .Void) {
                        return TypeResolutionError.TypeMismatch;
                    }
                }
            },
            .If => |*if_stmt| {
                // check both branches - use stmt pointer for scope key
                try self.validateReturnInBlockWithKey(&if_stmt.then_branch, expected_return_type, stmt, 0);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.validateReturnInBlockWithKey(else_branch, expected_return_type, stmt, 1);
                }
            },
            .While => |*while_stmt| {
                // check loop body
                try self.validateReturnInBlockWithKey(&while_stmt.body, expected_return_type, stmt, 0);
            },
            .ForLoop => |*for_stmt| {
                // check loop body
                try self.validateReturnInBlockWithKey(&for_stmt.body, expected_return_type, stmt, 0);
            },
            .LabeledBlock => |*labeled_block| {
                try self.validateReturnInBlockWithKey(&labeled_block.block, expected_return_type, stmt, 0);
            },
            .TryBlock => |*try_block| {
                try self.validateReturnInBlockWithKey(&try_block.try_block, expected_return_type, stmt, 0);
                if (try_block.catch_block) |*catch_block| {
                    try self.validateReturnInBlockWithKey(&catch_block.block, expected_return_type, stmt, 1);
                }
            },
            else => {
                // other statement types don't contain returns
            },
        }
    }

    fn functionEffectFromCore(
        allocator: std.mem.Allocator,
        eff: Effect,
    ) FunctionEffect {
        return switch (eff) {
            .Pure => FunctionEffect.pure(),
            .Reads => |slots| blk: {
                var list = std.ArrayList([]const u8){};
                for (slots.slots.items) |slot| {
                    list.append(allocator, slot) catch {};
                }
                break :blk FunctionEffect.reads(list);
            },
            .Writes => |slots| blk: {
                var list = std.ArrayList([]const u8){};
                for (slots.slots.items) |slot| {
                    list.append(allocator, slot) catch {};
                }
                break :blk FunctionEffect.writes(list);
            },
            .ReadsWrites => |rw| blk: {
                var reads = std.ArrayList([]const u8){};
                for (rw.reads.slots.items) |slot| {
                    reads.append(allocator, slot) catch {};
                }
                var writes = std.ArrayList([]const u8){};
                for (rw.writes.slots.items) |slot| {
                    writes.append(allocator, slot) catch {};
                }
                break :blk FunctionEffect.readsWrites(reads, writes);
            },
        };
    }

    /// Validate return statements in a block using statement-based key
    fn validateReturnInBlockWithKey(
        self: *TypeResolver,
        block: *Statements.BlockNode,
        expected_return_type: ?TypeInfo,
        parent_stmt: *Statements.StmtNode,
        block_id: usize,
    ) TypeResolutionError!void {
        // set scope for this block using stmt-based key (matches locals_binder)
        const prev_scope = self.current_scope;
        const block_key: usize = @intFromPtr(parent_stmt) * 4 + block_id;
        if (self.symbol_table.block_scopes.get(block_key)) |block_scope| {
            self.current_scope = block_scope;
            self.core_resolver.current_scope = block_scope;
        }
        defer {
            self.current_scope = prev_scope;
            self.core_resolver.current_scope = prev_scope;
        }

        for (block.statements) |*stmt| {
            try self.validateReturnInStatement(stmt, expected_return_type);
        }
    }

    /// Validate return statements in a block (legacy - for function body)
    fn validateReturnInBlock(
        self: *TypeResolver,
        block: *Statements.BlockNode,
        expected_return_type: ?TypeInfo,
    ) TypeResolutionError!void {
        // set scope for this block if it exists (function body uses block pointer)
        const prev_scope = self.current_scope;
        const block_key: usize = @intFromPtr(block);
        if (self.symbol_table.block_scopes.get(block_key)) |block_scope| {
            self.current_scope = block_scope;
            self.core_resolver.current_scope = block_scope;
        }
        defer {
            self.current_scope = prev_scope;
            self.core_resolver.current_scope = prev_scope;
        }

        for (block.statements) |*stmt| {
            try self.validateReturnInStatement(stmt, expected_return_type);
        }
    }

    // note: resolveTopLevelVariable removed - handled in resolveNodeTypes

    // ============================================================================
    // function Registry
    // ============================================================================

    fn registerFunctions(self: *TypeResolver, node: *AstNode) TypeResolutionError!void {
        switch (node.*) {
            .Function => |*function| {
                try self.function_registry.put(function.name, function);
            },
            .Contract => |*contract| {
                for (contract.body) |*child| {
                    if (child.* == .Function) {
                        try self.function_registry.put(child.Function.name, &child.Function);
                    }
                }
            },
            else => {},
        }
    }
};
