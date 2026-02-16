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
const comptime_fold = @import("comptime_fold.zig");
const monomorphize = @import("monomorphize.zig");
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
    GenericContractNotSupported,
    TopLevelGenericInstantiationNotSupported,
    /// Write to a slot that is currently locked (between @lock and @unlock)
    WriteToLockedSlot,
    /// @lock/@unlock only apply to storage slots
    LockUnlockOnlyStorage,
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
    // generic struct registry for monomorphization
    generic_structs: std.StringHashMap(*ast.StructDeclNode),
    // monomorphizer (created during resolveTypes, available for struct monomorphization)
    monomorphizer: ?*monomorphize.Monomorphizer = null,

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
            .generic_structs = std.StringHashMap(*ast.StructDeclNode).init(allocator),
        };
    }

    pub fn deinit(self: *TypeResolver) void {
        self.function_registry.deinit();
        self.generic_structs.deinit();
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

        // Create monomorphizer early so struct monomorphization can happen during type resolution
        var mono = monomorphize.Monomorphizer.init(self.type_storage_allocator);
        defer mono.deinit();
        self.monomorphizer = &mono;
        defer {
            self.monomorphizer = null;
        }

        // second pass: Resolve types (may trigger struct monomorphization)
        for (nodes) |*node| {
            try self.resolveNodeTypes(node, core.TypeContext{});
        }

        var fold_ctx = comptime_fold.FoldContext{
            .allocator = self.allocator,
            .type_storage_allocator = self.type_storage_allocator,
            .symbol_table = self.symbol_table,
            .core_resolver = &self.core_resolver,
            .current_scope = self.current_scope,
            .monomorphizer = &mono,
            .function_registry = &self.function_registry,
        };
        try comptime_fold.foldConstants(&fold_ctx, nodes);

        var injected_struct_count: usize = 0;
        var injected_function_count: usize = 0;

        // Keep integrating new monomorphized instances until no new ones appear.
        while (true) {
            const struct_instances = mono.getStructInstances();
            const function_instances = mono.getInstances();
            const new_struct_instances = struct_instances[injected_struct_count..];
            const new_function_instances = function_instances[injected_function_count..];

            if (new_struct_instances.len == 0 and new_function_instances.len == 0) break;

            if (new_struct_instances.len > 0) {
                try injectMonomorphizedStructs(self.type_storage_allocator, nodes, new_struct_instances, &mono);
            }
            if (new_function_instances.len > 0) {
                try injectMonomorphizedFunctions(self.type_storage_allocator, nodes, new_function_instances, &mono);
                try resolveMonomorphizedFunctions(self, nodes, new_function_instances);
            }

            injected_struct_count = struct_instances.len;
            injected_function_count = function_instances.len;

            // New instances can introduce fresh generic call sites.
            try comptime_fold.foldConstants(&fold_ctx, nodes);
        }
    }

    /// Inject monomorphized structs into contract bodies
    fn injectMonomorphizedStructs(
        alloc: std.mem.Allocator,
        nodes: []AstNode,
        struct_instances: []const ast.StructDeclNode,
        mono: *const monomorphize.Monomorphizer,
    ) TypeResolutionError!void {
        var injected_any = std.ArrayList(bool){};
        defer injected_any.deinit(alloc);
        try injected_any.resize(alloc, struct_instances.len);
        for (injected_any.items) |*v| v.* = false;

        for (nodes) |*node| {
            switch (node.*) {
                .Contract => |*contract| {
                    var inject_count: usize = 0;
                    for (struct_instances) |inst| {
                        if (shouldInjectStructIntoContract(contract, inst.name, mono)) {
                            inject_count += 1;
                        }
                    }
                    if (inject_count == 0) continue;

                    const new_body = try alloc.alloc(AstNode, contract.body.len + inject_count);
                    // Insert structs at the beginning (before functions that may use them)
                    var write_idx: usize = 0;
                    for (struct_instances, 0..) |inst, i| {
                        if (shouldInjectStructIntoContract(contract, inst.name, mono)) {
                            new_body[write_idx] = AstNode{ .StructDecl = inst };
                            write_idx += 1;
                            injected_any.items[i] = true;
                        }
                    }
                    @memcpy(new_body[write_idx..], contract.body);
                    contract.body = new_body;
                },
                else => {},
            }
        }

        for (injected_any.items, 0..) |ok, i| {
            if (!ok) {
                if (mono.getStructOwner(struct_instances[i].name) == null) {
                    log.err(
                        "[type_resolver] top-level generic struct instantiation '{s}' is not supported yet\n",
                        .{struct_instances[i].name},
                    );
                    return TypeResolutionError.TopLevelGenericInstantiationNotSupported;
                }
                log.err(
                    "[type_resolver] failed to inject monomorphized struct '{s}' into an owning contract\n",
                    .{struct_instances[i].name},
                );
                return TypeResolutionError.TypeMismatch;
            }
        }
    }

    /// Inject monomorphized functions into contract bodies
    fn injectMonomorphizedFunctions(
        alloc: std.mem.Allocator,
        nodes: []AstNode,
        instances: []const ast.FunctionNode,
        mono: *const monomorphize.Monomorphizer,
    ) TypeResolutionError!void {
        var injected_any = std.ArrayList(bool){};
        defer injected_any.deinit(alloc);
        try injected_any.resize(alloc, instances.len);
        for (injected_any.items) |*v| v.* = false;

        for (nodes) |*node| {
            switch (node.*) {
                .Contract => |*contract| {
                    var inject_count: usize = 0;
                    for (instances) |inst| {
                        if (shouldInjectFunctionIntoContract(contract, inst.name, mono)) {
                            inject_count += 1;
                        }
                    }
                    if (inject_count == 0) continue;

                    // Extend contract body with monomorphized functions
                    const new_body = try alloc.alloc(AstNode, contract.body.len + inject_count);
                    @memcpy(new_body[0..contract.body.len], contract.body);
                    var write_idx: usize = contract.body.len;
                    for (instances, 0..) |inst, i| {
                        if (shouldInjectFunctionIntoContract(contract, inst.name, mono)) {
                            new_body[write_idx] = AstNode{ .Function = inst };
                            write_idx += 1;
                            injected_any.items[i] = true;
                        }
                    }
                    contract.body = new_body;
                },
                else => {},
            }
        }

        for (injected_any.items, 0..) |ok, i| {
            if (!ok) {
                if (mono.getFunctionOwner(instances[i].name) == null) {
                    log.err(
                        "[type_resolver] top-level generic function instantiation '{s}' is not supported yet\n",
                        .{instances[i].name},
                    );
                    return TypeResolutionError.TopLevelGenericInstantiationNotSupported;
                }
                log.err(
                    "[type_resolver] failed to inject monomorphized function '{s}' into an owning contract\n",
                    .{instances[i].name},
                );
                return TypeResolutionError.TypeMismatch;
            }
        }
    }

    fn shouldInjectFunctionIntoContract(
        contract: *ContractNode,
        instance_name: []const u8,
        mono: *const monomorphize.Monomorphizer,
    ) bool {
        const owner = mono.getFunctionOwner(instance_name) orelse return false;
        return std.mem.eql(u8, owner, contract.name);
    }

    fn shouldInjectStructIntoContract(
        contract: *ContractNode,
        instance_name: []const u8,
        mono: *const monomorphize.Monomorphizer,
    ) bool {
        const owner = mono.getStructOwner(instance_name) orelse return false;
        return std.mem.eql(u8, owner, contract.name);
    }

    fn resolveMonomorphizedFunctions(
        self: *TypeResolver,
        nodes: []AstNode,
        instances: []const ast.FunctionNode,
    ) TypeResolutionError!void {
        for (nodes) |*node| {
            switch (node.*) {
                .Contract => |*contract| {
                    for (contract.body) |*child| {
                        switch (child.*) {
                            .Function => |*func| {
                                if (func.is_generic or func.is_comptime_only) continue;
                                for (instances) |inst| {
                                    if (std.mem.eql(u8, func.name, inst.name)) {
                                        try self.resolveFunction(func, core.TypeContext{});
                                        break;
                                    }
                                }
                            },
                            else => {},
                        }
                    }
                },
                else => {},
            }
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
                // Resolve generic struct types before core resolver processes it
                try self.tryResolveGenericStructType(&var_decl.type_info);
                if (var_decl.value) |value| {
                    try self.rewriteGenericStructInstantiationsExpr(value);
                }
                // convert to StmtNode for statement resolver
                var stmt_node = Statements.StmtNode{ .VariableDecl = var_decl.* };
                var typed = try self.core_resolver.resolveStatement(&stmt_node, context);
                defer typed.deinit(self.allocator);
                // write back resolved type_info to the original AST node
                var_decl.type_info = stmt_node.VariableDecl.type_info;
            },
            .Constant => |*constant| {
                try self.resolveConstant(constant, context);
            },
            .ErrorDecl => |*error_decl| {
                try self.resolveErrorDecl(error_decl, context);
            },
            .StructDecl => |*struct_decl| {
                try self.resolveStructDecl(struct_decl);
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

    /// If a TypeInfo has generic_type_args and refers to a generic struct,
    /// monomorphize the struct and rewrite the type to the mangled name.
    fn tryResolveGenericStructType(self: *TypeResolver, type_info: *TypeInfo) TypeResolutionError!void {
        const type_args = type_info.generic_type_args orelse return;
        if (type_args.len == 0) return;
        const ot = type_info.ora_type orelse return;
        if (ot != .struct_type) return;

        const struct_name = ot.struct_type;
        const mangled = (try self.tryMonomorphizeGenericStructByName(struct_name, type_args)) orelse return;

        // Rewrite type_info to concrete mangled name
        type_info.ora_type = OraType{ .struct_type = mangled };
        type_info.generic_type_args = null;
    }

    /// Detect generic struct declarations and mark field types with type_parameter.
    fn resolveStructDecl(self: *TypeResolver, decl: *ast.StructDeclNode) TypeResolutionError!void {
        // Detect comptime type parameters from parser
        if (decl.type_param_names.len > 0) {
            decl.is_generic = true;
            // Substitute struct_type references that match type param names
            for (decl.fields) |*field| {
                if (field.type_info.ora_type) |ot| {
                    if (ot == .struct_type) {
                        if (isTypeParamName(decl.type_param_names, ot.struct_type)) {
                            field.type_info.ora_type = OraType{ .type_parameter = ot.struct_type };
                            field.type_info.category = .Type;
                        }
                    }
                }
            }
        }
        // Register generic structs for monomorphization lookup
        if (decl.is_generic) {
            try self.generic_structs.put(decl.name, decl);
        }
    }

    fn resolveConstant(
        self: *TypeResolver,
        constant: *ast.ConstantNode,
        _: core.TypeContext,
    ) TypeResolutionError!void {
        try self.rewriteGenericStructInstantiationsExpr(constant.value);
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
        var indexed_count: usize = 0;
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
            if (field.indexed) {
                indexed_count += 1;
                switch (field.type_info.category) {
                    .Integer, .String, .Bool, .Address, .Bytes, .Hex => {},
                    else => return TypeResolutionError.TypeMismatch,
                }
            }
        }
        if (indexed_count > 3) {
            return TypeResolutionError.TypeMismatch;
        }
    }

    fn resolveContract(
        self: *TypeResolver,
        contract: *ContractNode,
        context: core.TypeContext,
    ) TypeResolutionError!void {
        if (contract.is_generic) {
            return TypeResolutionError.GenericContractNotSupported;
        }

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
        // --- Phase 0: detect comptime type parameters and mark generic ---
        var type_param_count: usize = 0;
        for (function.parameters) |param| {
            if (param.is_comptime) {
                if (param.type_info.ora_type) |ot| {
                    if (ot == .type) type_param_count += 1;
                }
            }
        }
        if (type_param_count > 0) {
            // Collect type param names
            const names = self.type_storage_allocator.alloc([]const u8, type_param_count) catch
                return TypeResolutionError.OutOfMemory;
            var idx: usize = 0;
            for (function.parameters) |param| {
                if (param.is_comptime) {
                    if (param.type_info.ora_type) |ot| {
                        if (ot == .type) {
                            names[idx] = param.name;
                            idx += 1;
                        }
                    }
                }
            }
            function.is_generic = true;
            function.type_param_names = names;
        }

        // --- Phase 1: resolve parameter types ---
        for (function.parameters) |*param| {
            // Try resolving generic struct types (e.g., Pair(u256))
            try self.tryResolveGenericStructType(&param.type_info);

            if (param.type_info.ora_type) |ot| {
                if (ot == .struct_type) {
                    const type_name = ot.struct_type;
                    // Check if this is a type parameter reference (e.g., `a: T`)
                    if (isTypeParamName(function.type_param_names, type_name)) {
                        param.type_info.ora_type = OraType{ .type_parameter = type_name };
                        param.type_info.category = .Type;
                        continue; // skip further resolution for this param
                    }
                    const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
                    const type_symbol = SymbolTable.findUp(root_scope, type_name);
                    if (type_symbol) |tsym| {
                        if (tsym.kind == .Enum) {
                            param.type_info.ora_type = OraType{ .enum_type = type_name };
                            param.type_info.category = .Enum;
                        } else if (tsym.kind == .Bitfield) {
                            param.type_info.ora_type = OraType{ .bitfield_type = type_name };
                            param.type_info.category = .Bitfield;
                        } else {
                            param.type_info.category = derivedCategory(ot);
                        }
                    } else {
                        param.type_info.category = derivedCategory(ot);
                    }
                } else if (ot == .type) {
                    // comptime T: type — already resolved
                    continue;
                } else {
                    param.type_info.category = derivedCategory(ot);
                }
            }
            if (!param.type_info.isResolved()) {
                return TypeResolutionError.UnresolvedType;
            }
            try self.core_resolver.validateErrorUnionType(param.type_info);
        }

        // --- Phase 2: resolve return type ---
        if (function.return_type_info) |*ret_type| {
            try self.tryResolveGenericStructType(ret_type);
            if (ret_type.ora_type) |ot| {
                if (ot == .struct_type) {
                    const type_name = ot.struct_type;
                    // Check if return type references a type parameter
                    if (isTypeParamName(function.type_param_names, type_name)) {
                        ret_type.ora_type = OraType{ .type_parameter = type_name };
                        ret_type.category = .Type;
                    } else {
                        const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
                        const type_symbol = SymbolTable.findUp(root_scope, type_name);
                        if (type_symbol) |tsym| {
                            if (tsym.kind == .Enum) {
                                ret_type.ora_type = OraType{ .enum_type = type_name };
                                ret_type.category = .Enum;
                            } else if (tsym.kind == .Bitfield) {
                                ret_type.ora_type = OraType{ .bitfield_type = type_name };
                                ret_type.category = .Bitfield;
                            } else {
                                ret_type.category = derivedCategory(ot);
                            }
                        } else {
                            ret_type.category = derivedCategory(ot);
                        }
                    }
                } else {
                    ret_type.category = derivedCategory(ot);
                }
            }
            if (!ret_type.isResolved() and !hasTypeParam(ret_type.ora_type)) {
                return TypeResolutionError.UnresolvedType;
            }
            if (!hasTypeParam(ret_type.ora_type)) {
                try self.core_resolver.validateErrorUnionType(ret_type.*);
            }
        }

        // Generic functions: skip body resolution (happens after monomorphization)
        if (function.is_generic) {
            return;
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

        // Push a comptime env scope so function-local consts don't leak to other functions
        self.core_resolver.comptime_env.pushScope(false) catch {};
        defer self.core_resolver.comptime_env.popScope();

        // Rewrite generic struct instantiations (e.g., Pair(u256) { ... }) before
        // statement-level type synthesis so struct field lookup sees concrete names.
        try self.rewriteGenericStructInstantiationsInFunction(function);

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
        self.core_resolver.clearLockedSlots(); // @lock/@unlock are tx-scoped at runtime.
        self.core_resolver.clearLockedBasesForFunction();
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

        // Persist locked storage roots for this function (for selective runtime guard emission).
        var bases_copy = std.StringHashMap(void).init(self.allocator);
        var bit = self.core_resolver.locked_bases_this_function.keyIterator();
        while (bit.next()) |k| {
            bases_copy.put(self.allocator.dupe(u8, k.*) catch return TypeResolutionError.OutOfMemory, {}) catch {};
        }
        if (self.symbol_table.function_locked_storage_roots.getPtr(function.name)) |existing| {
            var kit = existing.keyIterator();
            while (kit.next()) |kk| self.allocator.free(kk.*);
            existing.deinit();
        }
        self.symbol_table.function_locked_storage_roots.put(function.name, bases_copy) catch {};

        const stored_effect = functionEffectFromCore(self.allocator, &func_effect);
        if (self.symbol_table.function_effects.getPtr(function.name)) |existing| {
            existing.deinit(self.allocator);
            existing.* = stored_effect;
        } else {
            try self.symbol_table.function_effects.put(function.name, stored_effect);
        }

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
        eff: *Effect,
    ) FunctionEffect {
        const result = switch (eff.*) {
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
        eff.deinit(allocator);
        eff.* = Effect.pure();
        return result;
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

    // ============================================================================
    // Generics helpers
    // ============================================================================

    fn tryMonomorphizeGenericStructByName(
        self: *TypeResolver,
        struct_name: []const u8,
        type_args: []const OraType,
    ) TypeResolutionError!?[]const u8 {
        const generic_struct = self.generic_structs.get(struct_name) orelse return null;
        const mono_ptr = self.monomorphizer orelse return null;

        const owner_contract = self.symbol_table.findEnclosingContractName(self.current_scope);
        const mangled = try mono_ptr.monomorphizeStruct(generic_struct, type_args, owner_contract);
        try self.registerMonomorphizedStructSymbol(mangled, generic_struct.span, mono_ptr);
        return mangled;
    }

    fn registerMonomorphizedStructSymbol(
        self: *TypeResolver,
        mangled: []const u8,
        generic_span: SourceSpan,
        mono_ptr: *const monomorphize.Monomorphizer,
    ) TypeResolutionError!void {
        if (self.symbol_table.struct_fields.get(mangled) == null) {
            const mono_struct = mono_ptr.struct_cache.get(mangled) orelse return TypeResolutionError.TypeMismatch;
            try self.symbol_table.struct_fields.put(mangled, mono_struct.fields);

            const root = self.symbol_table.root;
            const sym = semantics.state.Symbol{
                .name = mangled,
                .kind = .Struct,
                .typ = TypeInfo.explicit(.Struct, OraType{ .struct_type = mangled }, generic_span),
                .span = generic_span,
            };
            _ = self.symbol_table.declare(root, sym) catch {};
        }
    }

    fn parseTypeArgsFromGenericCall(
        self: *TypeResolver,
        call: *ast.Expressions.CallExpr,
    ) TypeResolutionError![]OraType {
        if (call.arguments.len == 0) return TypeResolutionError.TypeMismatch;

        var args = std.ArrayList(OraType){};
        defer args.deinit(self.allocator);

        for (call.arguments) |arg| {
            const ora_ty = monomorphize.resolveTypeFromExpr(arg) orelse return TypeResolutionError.TypeMismatch;
            try args.append(self.allocator, ora_ty);
        }

        const owned = self.allocator.alloc(OraType, args.items.len) catch return TypeResolutionError.OutOfMemory;
        @memcpy(owned, args.items);
        return owned;
    }

    fn rewriteGenericStructInstantiationsInFunction(
        self: *TypeResolver,
        function: *FunctionNode,
    ) TypeResolutionError!void {
        for (function.parameters) |*param| {
            if (param.default_value) |default_expr| {
                try self.rewriteGenericStructInstantiationsExpr(default_expr);
            }
        }
        for (function.requires_clauses) |expr| {
            try self.rewriteGenericStructInstantiationsExpr(expr);
        }
        for (function.ensures_clauses) |expr| {
            try self.rewriteGenericStructInstantiationsExpr(expr);
        }
        try self.rewriteGenericStructInstantiationsBlock(&function.body);
    }

    fn rewriteGenericStructInstantiationsBlock(
        self: *TypeResolver,
        block: *Statements.BlockNode,
    ) TypeResolutionError!void {
        for (block.statements) |*stmt| {
            try self.rewriteGenericStructInstantiationsStmt(stmt);
        }
    }

    fn rewriteGenericStructInstantiationsSwitchPattern(
        self: *TypeResolver,
        pattern: *ast.Expressions.SwitchPattern,
    ) TypeResolutionError!void {
        switch (pattern.*) {
            .Range => |*range| {
                try self.rewriteGenericStructInstantiationsExpr(range.start);
                try self.rewriteGenericStructInstantiationsExpr(range.end);
            },
            else => {},
        }
    }

    fn rewriteGenericStructInstantiationsStmt(
        self: *TypeResolver,
        stmt: *Statements.StmtNode,
    ) TypeResolutionError!void {
        switch (stmt.*) {
            .Expr => |*expr| try self.rewriteGenericStructInstantiationsExpr(expr),
            .VariableDecl => |*var_decl| {
                try self.tryResolveGenericStructType(&var_decl.type_info);
                if (var_decl.value) |value| {
                    try self.rewriteGenericStructInstantiationsExpr(value);
                }
            },
            .DestructuringAssignment => |*destructuring| {
                try self.rewriteGenericStructInstantiationsExpr(destructuring.value);
            },
            .Return => |*ret| {
                if (ret.value) |*value| {
                    try self.rewriteGenericStructInstantiationsExpr(value);
                }
            },
            .If => |*if_stmt| {
                try self.rewriteGenericStructInstantiationsExpr(&if_stmt.condition);
                try self.rewriteGenericStructInstantiationsBlock(&if_stmt.then_branch);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.rewriteGenericStructInstantiationsBlock(else_branch);
                }
            },
            .While => |*while_stmt| {
                try self.rewriteGenericStructInstantiationsExpr(&while_stmt.condition);
                for (while_stmt.invariants) |*inv| {
                    try self.rewriteGenericStructInstantiationsExpr(inv);
                }
                if (while_stmt.decreases) |dec| {
                    try self.rewriteGenericStructInstantiationsExpr(dec);
                }
                if (while_stmt.increases) |inc| {
                    try self.rewriteGenericStructInstantiationsExpr(inc);
                }
                try self.rewriteGenericStructInstantiationsBlock(&while_stmt.body);
            },
            .ForLoop => |*for_stmt| {
                try self.rewriteGenericStructInstantiationsExpr(&for_stmt.iterable);
                for (for_stmt.invariants) |*inv| {
                    try self.rewriteGenericStructInstantiationsExpr(inv);
                }
                if (for_stmt.decreases) |dec| {
                    try self.rewriteGenericStructInstantiationsExpr(dec);
                }
                if (for_stmt.increases) |inc| {
                    try self.rewriteGenericStructInstantiationsExpr(inc);
                }
                try self.rewriteGenericStructInstantiationsBlock(&for_stmt.body);
            },
            .Break => |*break_stmt| {
                if (break_stmt.value) |value| {
                    try self.rewriteGenericStructInstantiationsExpr(value);
                }
            },
            .Continue => |*continue_stmt| {
                if (continue_stmt.value) |value| {
                    try self.rewriteGenericStructInstantiationsExpr(value);
                }
            },
            .Log => |*log_stmt| {
                for (log_stmt.args) |*arg| {
                    try self.rewriteGenericStructInstantiationsExpr(arg);
                }
            },
            .Lock => |*lock_stmt| try self.rewriteGenericStructInstantiationsExpr(&lock_stmt.path),
            .Unlock => |*unlock_stmt| try self.rewriteGenericStructInstantiationsExpr(&unlock_stmt.path),
            .Assert => |*assert_stmt| try self.rewriteGenericStructInstantiationsExpr(&assert_stmt.condition),
            .Invariant => |*inv| try self.rewriteGenericStructInstantiationsExpr(&inv.condition),
            .Requires => |*req| try self.rewriteGenericStructInstantiationsExpr(&req.condition),
            .Ensures => |*ens| try self.rewriteGenericStructInstantiationsExpr(&ens.condition),
            .Assume => |*assume| try self.rewriteGenericStructInstantiationsExpr(&assume.condition),
            .Switch => |*switch_stmt| {
                try self.rewriteGenericStructInstantiationsExpr(&switch_stmt.condition);
                for (switch_stmt.cases) |*case| {
                    try self.rewriteGenericStructInstantiationsSwitchPattern(&case.pattern);
                    switch (case.body) {
                        .Expression => |expr| try self.rewriteGenericStructInstantiationsExpr(expr),
                        .Block => |*case_block| try self.rewriteGenericStructInstantiationsBlock(case_block),
                        .LabeledBlock => |*labeled| try self.rewriteGenericStructInstantiationsBlock(&labeled.block),
                    }
                }
                if (switch_stmt.default_case) |*default_block| {
                    try self.rewriteGenericStructInstantiationsBlock(default_block);
                }
            },
            .LabeledBlock => |*labeled| {
                try self.rewriteGenericStructInstantiationsBlock(&labeled.block);
            },
            .ErrorDecl => |*error_decl| {
                if (error_decl.parameters) |params| {
                    for (params) |*param| {
                        try self.tryResolveGenericStructType(&param.type_info);
                        if (param.default_value) |default_val| {
                            try self.rewriteGenericStructInstantiationsExpr(default_val);
                        }
                    }
                }
            },
            .TryBlock => |*try_block| {
                try self.rewriteGenericStructInstantiationsBlock(&try_block.try_block);
                if (try_block.catch_block) |*catch_block| {
                    try self.rewriteGenericStructInstantiationsBlock(&catch_block.block);
                }
            },
            .CompoundAssignment => |*compound| {
                try self.rewriteGenericStructInstantiationsExpr(compound.target);
                try self.rewriteGenericStructInstantiationsExpr(compound.value);
            },
            else => {},
        }
    }

    fn rewriteGenericStructInstantiationsExpr(
        self: *TypeResolver,
        expr: *ast.Expressions.ExprNode,
    ) TypeResolutionError!void {
        switch (expr.*) {
            .Binary => |*binary| {
                try self.rewriteGenericStructInstantiationsExpr(binary.lhs);
                try self.rewriteGenericStructInstantiationsExpr(binary.rhs);
            },
            .Unary => |*unary| {
                try self.rewriteGenericStructInstantiationsExpr(unary.operand);
            },
            .Assignment => |*assignment| {
                try self.rewriteGenericStructInstantiationsExpr(assignment.target);
                try self.rewriteGenericStructInstantiationsExpr(assignment.value);
            },
            .CompoundAssignment => |*compound| {
                try self.rewriteGenericStructInstantiationsExpr(compound.target);
                try self.rewriteGenericStructInstantiationsExpr(compound.value);
            },
            .Call => |*call| {
                try self.rewriteGenericStructInstantiationsExpr(call.callee);
                for (call.arguments) |arg| {
                    try self.rewriteGenericStructInstantiationsExpr(arg);
                }
            },
            .Index => |*index_expr| {
                try self.rewriteGenericStructInstantiationsExpr(index_expr.target);
                try self.rewriteGenericStructInstantiationsExpr(index_expr.index);
            },
            .FieldAccess => |*field_access| {
                try self.rewriteGenericStructInstantiationsExpr(field_access.target);
            },
            .Cast => |*cast_expr| {
                try self.tryResolveGenericStructType(&cast_expr.target_type);
                try self.rewriteGenericStructInstantiationsExpr(cast_expr.operand);
            },
            .Comptime => |*comptime_expr| {
                try self.rewriteGenericStructInstantiationsBlock(&comptime_expr.block);
            },
            .Old => |*old_expr| {
                try self.rewriteGenericStructInstantiationsExpr(old_expr.expr);
            },
            .Quantified => |*quantified| {
                try self.tryResolveGenericStructType(&quantified.variable_type);
                if (quantified.condition) |condition| {
                    try self.rewriteGenericStructInstantiationsExpr(condition);
                }
                try self.rewriteGenericStructInstantiationsExpr(quantified.body);
            },
            .Tuple => |*tuple_expr| {
                for (tuple_expr.elements) |element| {
                    try self.rewriteGenericStructInstantiationsExpr(element);
                }
            },
            .SwitchExpression => |*switch_expr| {
                try self.rewriteGenericStructInstantiationsExpr(switch_expr.condition);
                for (switch_expr.cases) |*case| {
                    try self.rewriteGenericStructInstantiationsSwitchPattern(&case.pattern);
                    switch (case.body) {
                        .Expression => |case_expr| try self.rewriteGenericStructInstantiationsExpr(case_expr),
                        .Block => |*case_block| try self.rewriteGenericStructInstantiationsBlock(case_block),
                        .LabeledBlock => |*labeled| try self.rewriteGenericStructInstantiationsBlock(&labeled.block),
                    }
                }
                if (switch_expr.default_case) |*default_case| {
                    try self.rewriteGenericStructInstantiationsBlock(default_case);
                }
            },
            .Try => |*try_expr| {
                try self.rewriteGenericStructInstantiationsExpr(try_expr.expr);
            },
            .ErrorCast => |*error_cast| {
                try self.tryResolveGenericStructType(&error_cast.target_type);
                try self.rewriteGenericStructInstantiationsExpr(error_cast.operand);
            },
            .Shift => |*shift_expr| {
                try self.rewriteGenericStructInstantiationsExpr(shift_expr.mapping);
                try self.rewriteGenericStructInstantiationsExpr(shift_expr.source);
                try self.rewriteGenericStructInstantiationsExpr(shift_expr.dest);
                try self.rewriteGenericStructInstantiationsExpr(shift_expr.amount);
            },
            .StructInstantiation => |*struct_inst| {
                for (struct_inst.fields) |*field| {
                    try self.rewriteGenericStructInstantiationsExpr(field.value);
                }

                if (struct_inst.struct_name.* == .Call) {
                    const call = &struct_inst.struct_name.Call;
                    if (call.callee.* == .Identifier) {
                        const type_args = try self.parseTypeArgsFromGenericCall(call);
                        defer self.allocator.free(type_args);
                        const base_name = call.callee.Identifier.name;
                        if (try self.tryMonomorphizeGenericStructByName(base_name, type_args)) |mangled| {
                            struct_inst.struct_name.* = ast.Expressions.ExprNode{
                                .Identifier = ast.Expressions.IdentifierExpr{
                                    .name = mangled,
                                    .type_info = TypeInfo.explicit(.Struct, OraType{ .struct_type = mangled }, call.span),
                                    .span = call.span,
                                },
                            };
                        }
                    }
                }

                try self.rewriteGenericStructInstantiationsExpr(struct_inst.struct_name);
            },
            .AnonymousStruct => |*anon_struct| {
                for (anon_struct.fields) |*field| {
                    try self.rewriteGenericStructInstantiationsExpr(field.value);
                }
            },
            .Range => |*range_expr| {
                try self.rewriteGenericStructInstantiationsExpr(range_expr.start);
                try self.rewriteGenericStructInstantiationsExpr(range_expr.end);
            },
            .LabeledBlock => |*labeled| {
                try self.rewriteGenericStructInstantiationsBlock(&labeled.block);
            },
            .Destructuring => |*destructuring| {
                try self.rewriteGenericStructInstantiationsExpr(destructuring.value);
            },
            .ArrayLiteral => |*array_literal| {
                for (array_literal.elements) |element| {
                    try self.rewriteGenericStructInstantiationsExpr(element);
                }
            },
            else => {},
        }
    }

    fn isTypeParamName(type_param_names: []const []const u8, name: []const u8) bool {
        for (type_param_names) |tp| {
            if (std.mem.eql(u8, tp, name)) return true;
        }
        return false;
    }

    fn hasTypeParam(ora_type: ?OraType) bool {
        if (ora_type) |ot| return ot == .type_parameter;
        return false;
    }

    fn derivedCategory(ot: OraType) @import("../type_info.zig").TypeCategory {
        var cat = ot.getCategory();
        if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
            cat = .ErrorUnion;
        }
        return cat;
    }
};
