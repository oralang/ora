const std = @import("std");
const lib = @import("ora_lib");
const import_resolver = @import("mod.zig");

pub const ResolverOptions = import_resolver.ResolverOptions;

pub const ExportKind = lib.semantics.state.ExportKind;
pub const ModuleExportMap = lib.semantics.state.ModuleExportMap;

pub const ParsedProgram = struct {
    nodes: []lib.AstNode,
    arena: lib.ast_arena.AstArena,
    module_exports: ?*ModuleExportMap = null,
    module_exports_allocator: ?std.mem.Allocator = null,

    pub fn deinit(self: *ParsedProgram) void {
        if (self.module_exports) |me| {
            me.deinit();
            if (self.module_exports_allocator) |me_allocator| {
                me_allocator.destroy(me);
            }
        }
        self.arena.deinit();
        self.* = undefined;
    }
};

const ParsedModuleState = struct {
    nodes: []lib.AstNode = &.{},
    exports: std.StringHashMap(ExportKind),
    alias_targets: std.StringHashMap([]const u8),

    fn init(allocator: std.mem.Allocator) ParsedModuleState {
        return .{
            .nodes = &.{},
            .exports = std.StringHashMap(ExportKind).init(allocator),
            .alias_targets = std.StringHashMap([]const u8).init(allocator),
        };
    }

    fn deinit(self: *ParsedModuleState) void {
        self.exports.deinit();
        self.alias_targets.deinit();
    }
};

fn injectImportedRuntimeFunctionsIntoEntryContracts(
    arena_allocator: std.mem.Allocator,
    entry_nodes: []lib.AstNode,
    imported_runtime_functions: []const lib.FunctionNode,
) !void {
    if (imported_runtime_functions.len == 0) return;

    for (entry_nodes) |*node| {
        if (node.* != .Contract) continue;

        const contract = &node.Contract;
        const existing_body = contract.body;
        const new_body = try arena_allocator.alloc(lib.AstNode, existing_body.len + imported_runtime_functions.len);
        @memcpy(new_body[0..existing_body.len], existing_body);

        var write_idx: usize = existing_body.len;
        for (imported_runtime_functions) |imported_fn| {
            var local_fn = imported_fn;
            // v1 importer: imported runtime functions are contract-internal helpers.
            local_fn.visibility = .Private;
            new_body[write_idx] = .{ .Function = local_fn };
            write_idx += 1;
        }
        contract.body = new_body;
    }
}

fn ensureLogSignaturesForProgram(symbols: *lib.semantics.state.SymbolTable, nodes: []const lib.AstNode) !void {
    for (nodes) |node| switch (node) {
        .LogDecl => |l| {
            if (symbols.log_signatures.get(l.name) == null) {
                try symbols.log_signatures.put(l.name, l.fields);
            }
        },
        .Contract => |c| {
            if (symbols.contract_log_signatures.getPtr(c.name) == null) {
                const log_map = std.StringHashMap([]const lib.ast.LogField).init(symbols.allocator);
                try symbols.contract_log_signatures.put(c.name, log_map);
            }
            for (c.body) |member| switch (member) {
                .LogDecl => |l| {
                    if (symbols.contract_log_signatures.getPtr(c.name)) |log_map| {
                        if (log_map.get(l.name) == null) {
                            try log_map.put(l.name, l.fields);
                        }
                    }
                },
                else => {},
            };
        },
        else => {},
    };
}

// ---------------------------------------------------------------------------
// EnumLiteral -> FieldAccess normalization
//
// The parser may represent `alias.member` as EnumLiteral when `alias` is not
// in a hardcoded module namespace list. This pass converts those back to
// FieldAccess so that downstream phases see uniform qualified access.
// ---------------------------------------------------------------------------

fn normalizeExprEnumLiterals(
    arena: *lib.ast_arena.AstArena,
    expr: *lib.ast.Expressions.ExprNode,
    alias_targets: *const std.StringHashMap([]const u8),
) anyerror!void {
    switch (expr.*) {
        .EnumLiteral => |el| {
            if (alias_targets.get(el.enum_name)) |_| {
                const target_ptr = try arena.createNode(lib.ast.Expressions.ExprNode);
                target_ptr.* = .{ .Identifier = .{
                    .name = try arena.createString(el.enum_name),
                    .type_info = lib.ast.Types.TypeInfo.unknown(),
                    .span = el.span,
                } };
                expr.* = .{ .FieldAccess = .{
                    .target = target_ptr,
                    .field = try arena.createString(el.variant_name),
                    .type_info = lib.ast.Types.TypeInfo.unknown(),
                    .span = el.span,
                } };
            }
        },
        .Identifier, .Literal, .ErrorReturn => {},
        .Quantified => |*q| {
            if (q.condition) |c| try normalizeExprEnumLiterals(arena, c, alias_targets);
            try normalizeExprEnumLiterals(arena, q.body, alias_targets);
        },
        .Unary => |*u| try normalizeExprEnumLiterals(arena, u.operand, alias_targets),
        .Binary => |*b| {
            try normalizeExprEnumLiterals(arena, b.lhs, alias_targets);
            try normalizeExprEnumLiterals(arena, b.rhs, alias_targets);
        },
        .Assignment => |*a| {
            try normalizeExprEnumLiterals(arena, a.target, alias_targets);
            try normalizeExprEnumLiterals(arena, a.value, alias_targets);
        },
        .CompoundAssignment => |*ca| {
            try normalizeExprEnumLiterals(arena, ca.target, alias_targets);
            try normalizeExprEnumLiterals(arena, ca.value, alias_targets);
        },
        .Call => |*c| {
            try normalizeExprEnumLiterals(arena, c.callee, alias_targets);
            for (c.arguments) |arg| try normalizeExprEnumLiterals(arena, arg, alias_targets);
        },
        .Index => |*ix| {
            try normalizeExprEnumLiterals(arena, ix.target, alias_targets);
            try normalizeExprEnumLiterals(arena, ix.index, alias_targets);
        },
        .FieldAccess => |*fa| try normalizeExprEnumLiterals(arena, fa.target, alias_targets),
        .Cast => |*c| try normalizeExprEnumLiterals(arena, c.operand, alias_targets),
        .Comptime => |*ct| {
            for (ct.block.statements) |*stmt| try normalizeStmtEnumLiterals(arena, stmt, alias_targets);
        },
        .Old => |*o| try normalizeExprEnumLiterals(arena, o.expr, alias_targets),
        .Tuple => |*t| {
            for (t.elements) |el| try normalizeExprEnumLiterals(arena, el, alias_targets);
        },
        .StructInstantiation => |*si| {
            try normalizeExprEnumLiterals(arena, si.struct_name, alias_targets);
            for (si.fields) |*f| try normalizeExprEnumLiterals(arena, f.value, alias_targets);
        },
        .AnonymousStruct => |*as| {
            for (as.fields) |*f| try normalizeExprEnumLiterals(arena, f.value, alias_targets);
        },
        .ArrayLiteral => |*al| {
            for (al.elements) |el| try normalizeExprEnumLiterals(arena, el, alias_targets);
        },
        .SwitchExpression => |*sw| {
            try normalizeExprEnumLiterals(arena, sw.condition, alias_targets);
            for (sw.cases) |*case| {
                switch (case.pattern) {
                    .Range => |r| {
                        try normalizeExprEnumLiterals(arena, r.start, alias_targets);
                        try normalizeExprEnumLiterals(arena, r.end, alias_targets);
                    },
                    else => {},
                }
                switch (case.body) {
                    .Expression => |e| try normalizeExprEnumLiterals(arena, e, alias_targets),
                    .Block => |*b| for (b.statements) |*stmt| try normalizeStmtEnumLiterals(arena, stmt, alias_targets),
                    .LabeledBlock => |*lb| for (lb.block.statements) |*stmt| try normalizeStmtEnumLiterals(arena, stmt, alias_targets),
                }
            }
            if (sw.default_case) |*db| {
                for (db.statements) |*stmt| try normalizeStmtEnumLiterals(arena, stmt, alias_targets);
            }
        },
        .Range => |*r| {
            try normalizeExprEnumLiterals(arena, r.start, alias_targets);
            try normalizeExprEnumLiterals(arena, r.end, alias_targets);
        },
        .LabeledBlock => |*lb| {
            for (lb.block.statements) |*stmt| try normalizeStmtEnumLiterals(arena, stmt, alias_targets);
        },
        .Destructuring => |*d| try normalizeExprEnumLiterals(arena, d.value, alias_targets),
        .Try => |*t| try normalizeExprEnumLiterals(arena, t.expr, alias_targets),
        .ErrorCast => |*ec| try normalizeExprEnumLiterals(arena, ec.operand, alias_targets),
        .Shift => |*sh| {
            try normalizeExprEnumLiterals(arena, sh.mapping, alias_targets);
            try normalizeExprEnumLiterals(arena, sh.source, alias_targets);
            try normalizeExprEnumLiterals(arena, sh.dest, alias_targets);
            try normalizeExprEnumLiterals(arena, sh.amount, alias_targets);
        },
    }
}

fn normalizeStmtEnumLiterals(
    arena: *lib.ast_arena.AstArena,
    stmt: *lib.ast.Statements.StmtNode,
    alias_targets: *const std.StringHashMap([]const u8),
) anyerror!void {
    switch (stmt.*) {
        .Expr => |*expr| try normalizeExprEnumLiterals(arena, expr, alias_targets),
        .VariableDecl => |*v| {
            if (v.value) |value| try normalizeExprEnumLiterals(arena, value, alias_targets);
        },
        .DestructuringAssignment => |*d| try normalizeExprEnumLiterals(arena, d.value, alias_targets),
        .Return => |*r| {
            if (r.value) |*value| try normalizeExprEnumLiterals(arena, value, alias_targets);
        },
        .If => |*if_stmt| {
            try normalizeExprEnumLiterals(arena, &if_stmt.condition, alias_targets);
            for (if_stmt.then_branch.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
            if (if_stmt.else_branch) |*else_branch| {
                for (else_branch.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
            }
        },
        .While => |*w| {
            try normalizeExprEnumLiterals(arena, &w.condition, alias_targets);
            for (w.invariants) |*inv| try normalizeExprEnumLiterals(arena, inv, alias_targets);
            if (w.decreases) |decreases| try normalizeExprEnumLiterals(arena, decreases, alias_targets);
            if (w.increases) |increases| try normalizeExprEnumLiterals(arena, increases, alias_targets);
            for (w.body.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
        },
        .ForLoop => |*fl| {
            try normalizeExprEnumLiterals(arena, &fl.iterable, alias_targets);
            for (fl.invariants) |*inv| try normalizeExprEnumLiterals(arena, inv, alias_targets);
            if (fl.decreases) |decreases| try normalizeExprEnumLiterals(arena, decreases, alias_targets);
            if (fl.increases) |increases| try normalizeExprEnumLiterals(arena, increases, alias_targets);
            for (fl.body.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
        },
        .Break => |*b| {
            if (b.value) |value| try normalizeExprEnumLiterals(arena, value, alias_targets);
        },
        .Continue => |*c| {
            if (c.value) |value| try normalizeExprEnumLiterals(arena, value, alias_targets);
        },
        .Log => |*l| {
            for (l.args) |*arg| try normalizeExprEnumLiterals(arena, arg, alias_targets);
        },
        .Lock => |*l| try normalizeExprEnumLiterals(arena, &l.path, alias_targets),
        .Unlock => |*u| try normalizeExprEnumLiterals(arena, &u.path, alias_targets),
        .Assert => |*a| try normalizeExprEnumLiterals(arena, &a.condition, alias_targets),
        .Invariant => |*i| try normalizeExprEnumLiterals(arena, &i.condition, alias_targets),
        .Requires => |*r| try normalizeExprEnumLiterals(arena, &r.condition, alias_targets),
        .Ensures => |*e| try normalizeExprEnumLiterals(arena, &e.condition, alias_targets),
        .Assume => |*a| try normalizeExprEnumLiterals(arena, &a.condition, alias_targets),
        .Havoc, .ErrorDecl => {},
        .Switch => |*sw| {
            try normalizeExprEnumLiterals(arena, &sw.condition, alias_targets);
            for (sw.cases) |*case| {
                switch (case.pattern) {
                    .Range => |r| {
                        try normalizeExprEnumLiterals(arena, r.start, alias_targets);
                        try normalizeExprEnumLiterals(arena, r.end, alias_targets);
                    },
                    else => {},
                }
                switch (case.body) {
                    .Expression => |e| try normalizeExprEnumLiterals(arena, e, alias_targets),
                    .Block => |*b| for (b.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets),
                    .LabeledBlock => |*lb| for (lb.block.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets),
                }
            }
            if (sw.default_case) |*db| {
                for (db.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
            }
        },
        .LabeledBlock => |*lb| {
            for (lb.block.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
        },
        .TryBlock => |*tb| {
            for (tb.try_block.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
            if (tb.catch_block) |*catch_block| {
                for (catch_block.block.statements) |*nested| try normalizeStmtEnumLiterals(arena, nested, alias_targets);
            }
        },
        .CompoundAssignment => |*ca| {
            try normalizeExprEnumLiterals(arena, ca.target, alias_targets);
            try normalizeExprEnumLiterals(arena, ca.value, alias_targets);
        },
    }
}

fn normalizeNodeEnumLiterals(
    arena: *lib.ast_arena.AstArena,
    node: *lib.AstNode,
    alias_targets: *const std.StringHashMap([]const u8),
) anyerror!void {
    switch (node.*) {
        .Contract => |*c| {
            for (c.body) |*member| try normalizeNodeEnumLiterals(arena, member, alias_targets);
        },
        .Function => |*f| {
            for (f.requires_clauses) |req| try normalizeExprEnumLiterals(arena, req, alias_targets);
            for (f.ensures_clauses) |ens| try normalizeExprEnumLiterals(arena, ens, alias_targets);
            for (f.body.statements) |*stmt| try normalizeStmtEnumLiterals(arena, stmt, alias_targets);
        },
        .Constant => |*cst| try normalizeExprEnumLiterals(arena, cst.value, alias_targets),
        .VariableDecl => |*v| {
            if (v.value) |value| try normalizeExprEnumLiterals(arena, value, alias_targets);
        },
        .ContractInvariant => |*inv| try normalizeExprEnumLiterals(arena, inv.condition, alias_targets),
        else => {},
    }
}

fn validateImportedLibraryModule(module_path: []const u8, nodes: []const lib.AstNode) !void {
    for (nodes) |node| {
        switch (node) {
            .Import, .Constant, .StructDecl, .BitfieldDecl => {},
            .Function => |f| {
                if (std.mem.eql(u8, f.name, "init")) {
                    std.log.warn("Imported library '{s}' cannot declare 'init' in v1 importer.", .{module_path});
                    return error.ImportedLibraryInitNotAllowed;
                }
                if (f.visibility != .Public and !f.is_comptime_only) {
                    std.log.warn("Imported library '{s}' has non-public runtime function '{s}'. v1 allows only 'pub fn' (or comptime-only).", .{
                        module_path, f.name,
                    });
                    return error.ImportedLibraryFunctionVisibilityNotAllowed;
                }
            },
            .VariableDecl => |v| {
                std.log.warn("Imported library '{s}' cannot declare top-level variable '{s}' in v1 importer.", .{
                    module_path, v.name,
                });
                return error.ImportedLibraryStateNotAllowed;
            },
            .Contract => |c| {
                std.log.warn("Imported library '{s}' contains unsupported contract '{s}' (left to v2 importer).", .{
                    module_path, c.name,
                });
                return error.ImportedLibraryDeclarationNotAllowed;
            },
            .EnumDecl => |e| {
                std.log.warn("Imported library '{s}' contains unsupported enum '{s}' (left to v2 importer).", .{
                    module_path, e.name,
                });
                return error.ImportedLibraryDeclarationNotAllowed;
            },
            .LogDecl => |l| {
                std.log.warn("Imported library '{s}' contains unsupported log '{s}' (left to v2 importer).", .{
                    module_path, l.name,
                });
                return error.ImportedLibraryDeclarationNotAllowed;
            },
            .ErrorDecl => |e| {
                std.log.warn("Imported library '{s}' contains unsupported error '{s}' (left to v2 importer).", .{
                    module_path, e.name,
                });
                return error.ImportedLibraryDeclarationNotAllowed;
            },
            .ContractInvariant => |inv| {
                std.log.warn("Imported library '{s}' contains unsupported contract invariant '{s}' (left to v2 importer).", .{
                    module_path, inv.name,
                });
                return error.ImportedLibraryDeclarationNotAllowed;
            },
            .Module, .Block, .Expression, .Statement, .TryBlock => {
                std.log.warn("Imported library '{s}' contains unsupported top-level node kind (left to v2 importer).", .{
                    module_path,
                });
                return error.ImportedLibraryDeclarationNotAllowed;
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Program loading
// ---------------------------------------------------------------------------

pub fn loadProgramWithImportsRawWithResolverOptions(
    allocator: std.mem.Allocator,
    entry_file_path: []const u8,
    resolver_options: ResolverOptions,
) !ParsedProgram {
    var graph = try import_resolver.resolveImportGraph(allocator, entry_file_path, resolver_options);
    defer graph.deinit(allocator);

    var arena = lib.ast_arena.AstArena.init(allocator);
    errdefer arena.deinit();

    const arena_allocator = arena.allocator();
    const module_count = graph.modules.len;

    var module_states = try allocator.alloc(ParsedModuleState, module_count);
    defer allocator.free(module_states);
    for (module_states) |*state| state.* = ParsedModuleState.init(allocator);
    defer for (module_states) |*state| state.deinit();

    var module_index_by_id = std.StringHashMap(usize).init(allocator);
    defer module_index_by_id.deinit();
    for (graph.modules, 0..) |module, i| {
        try module_index_by_id.put(module.canonical_id, i);
    }

    const entry_module_idx = module_index_by_id.get(graph.entry_canonical_id) orelse return error.ImportTargetNotFound;

    // Parse all modules and collect exports + alias mappings.
    for (graph.modules, 0..) |module, i| {
        // Build alias -> canonical_id mapping, enforcing unique aliases.
        for (module.imports) |resolved_import| {
            if (module_states[i].alias_targets.get(resolved_import.alias)) |existing| {
                if (!std.mem.eql(u8, existing, resolved_import.target_canonical_id)) {
                    std.log.warn("Duplicate import alias '{s}' in '{s}' maps to different modules.", .{
                        resolved_import.alias, module.resolved_path,
                    });
                    return error.DuplicateImportAlias;
                }
                continue;
            }
            try module_states[i].alias_targets.put(resolved_import.alias, resolved_import.target_canonical_id);
        }

        const source = std.fs.cwd().readFileAlloc(arena_allocator, module.resolved_path, 1024 * 1024) catch |err| {
            std.log.warn("Failed to read module '{s}': {s}", .{ module.resolved_path, @errorName(err) });
            return error.FileNotFound;
        };

        var lexer = lib.Lexer.init(arena_allocator, source);
        defer lexer.deinit();
        const tokens = try lexer.scanTokens();

        var parser = lib.Parser.init(tokens, &arena);
        parser.setFileId(@intCast(i + 1));
        const nodes = try parser.parse();
        module_states[i].nodes = nodes;

        const is_entry_module = std.mem.eql(u8, module.canonical_id, graph.entry_canonical_id);
        if (!is_entry_module) {
            try validateImportedLibraryModule(module.resolved_path, nodes);
        }

        for (nodes) |node| switch (node) {
            .Function => |f| try module_states[i].exports.put(f.name, .Function),
            .Constant => |c| try module_states[i].exports.put(c.name, .Constant),
            .StructDecl => |s| try module_states[i].exports.put(s.name, .StructDecl),
            .BitfieldDecl => |b| try module_states[i].exports.put(b.name, .BitfieldDecl),
            // Left this to v2 of importer:
            // .Contract => |c| try module_states[i].exports.put(c.name, .Contract),
            // .EnumDecl => |e| try module_states[i].exports.put(e.name, .EnumDecl),
            // .LogDecl => |l| try module_states[i].exports.put(l.name, .LogDecl),
            // .ErrorDecl => |e| try module_states[i].exports.put(e.name, .ErrorDecl),
            // .VariableDecl => |v| try module_states[i].exports.put(v.name, .Variable),
            else => {},
        };
    }

    // v1 importer behavior:
    // Imported runtime functions are materialized as private helpers inside
    // entry contracts so alias.fn() in contracts lowers to an in-contract
    // func.call target.
    var imported_runtime_functions = std.ArrayList(lib.FunctionNode){};
    defer imported_runtime_functions.deinit(arena_allocator);
    for (module_states, 0..) |state, i| {
        if (i == entry_module_idx) continue;
        for (state.nodes) |node| {
            if (node != .Function) continue;
            const f = node.Function;
            if (f.is_comptime_only) continue;
            try imported_runtime_functions.append(arena_allocator, f);
        }
    }
    try injectImportedRuntimeFunctionsIntoEntryContracts(
        arena_allocator,
        module_states[entry_module_idx].nodes,
        imported_runtime_functions.items,
    );

    var entry_has_contract = false;
    for (module_states[entry_module_idx].nodes) |node| {
        if (node == .Contract) {
            entry_has_contract = true;
            break;
        }
    }

    // Normalize EnumLiteral -> FieldAccess for known import aliases (parser quirk).
    for (module_states) |*state| {
        for (state.nodes) |*node| {
            try normalizeNodeEnumLiterals(&arena, node, &state.alias_targets);
        }
    }

    // Build the ModuleExportMap: alias -> { member -> ExportKind }.
    // Use the stable function allocator for hash-map internals, not the local
    // arena allocator (the arena value is moved on return).
    var mod_exports = try allocator.create(ModuleExportMap);
    errdefer allocator.destroy(mod_exports);
    mod_exports.* = ModuleExportMap.init(allocator);
    errdefer mod_exports.deinit();

    for (module_states) |state| {
        var alias_it = state.alias_targets.iterator();
        while (alias_it.next()) |alias_entry| {
            const alias = alias_entry.key_ptr.*;
            const target_id = alias_entry.value_ptr.*;
            const target_idx = module_index_by_id.get(target_id) orelse continue;

            if (mod_exports.entries.contains(alias)) continue;

            const alias_copy = try arena_allocator.dupe(u8, alias);
            var export_copy = std.StringHashMap(ExportKind).init(allocator);
            var src_it = module_states[target_idx].exports.iterator();
            while (src_it.next()) |ex| try export_copy.put(ex.key_ptr.*, ex.value_ptr.*);
            try mod_exports.entries.put(alias_copy, export_copy);
        }
    }

    // Merge all module nodes. Deduplicate Import nodes by alias so that the
    // same alias imported in multiple modules doesn't cause false redeclaration
    // errors in semantics collect.
    var merged_nodes = std.ArrayList(lib.AstNode){};
    defer merged_nodes.deinit(arena_allocator);
    var seen_import_aliases = std.StringHashMap(void).init(allocator);
    defer seen_import_aliases.deinit();

    for (module_states, 0..) |state, module_idx| {
        for (state.nodes) |node| {
            if (node == .Import) {
                const alias_name = node.Import.alias orelse node.Import.path;
                if (seen_import_aliases.contains(alias_name)) continue;
                try seen_import_aliases.put(alias_name, {});
            }
            if (entry_has_contract and module_idx != entry_module_idx and node == .Function) {
                const imported_fn = node.Function;
                if (!imported_fn.is_comptime_only) {
                    // v1 importer: runtime imported functions are injected into
                    // entry contracts; do not keep them as top-level module funcs.
                    continue;
                }
            }
            try merged_nodes.append(arena_allocator, node);
        }
    }

    const all_nodes = try merged_nodes.toOwnedSlice(arena_allocator);
    return .{
        .nodes = all_nodes,
        .arena = arena,
        .module_exports = mod_exports,
        .module_exports_allocator = allocator,
    };
}

pub fn loadProgramWithImportsRaw(allocator: std.mem.Allocator, entry_file_path: []const u8) !ParsedProgram {
    return loadProgramWithImportsRawWithResolverOptions(allocator, entry_file_path, .{});
}

pub fn loadProgramWithImportsTypedWithResolverOptions(
    allocator: std.mem.Allocator,
    entry_file_path: []const u8,
    resolver_options: ResolverOptions,
) !ParsedProgram {
    var program = try loadProgramWithImportsRawWithResolverOptions(allocator, entry_file_path, resolver_options);
    errdefer program.deinit();

    var semantics_result = try lib.semantics.core.analyzePhase1(allocator, program.nodes);
    defer allocator.free(semantics_result.diagnostics);
    defer semantics_result.symbols.deinit();

    if (semantics_result.diagnostics.len > 0) {
        for (semantics_result.diagnostics) |diag| {
            std.log.warn("{s} (line {d})", .{ diag.message, diag.span.line });
        }
        return error.RedeclarationConflict;
    }

    // Thread module export info into the symbol table for the type resolver.
    if (program.module_exports) |me| {
        semantics_result.symbols.module_exports = me;
    }

    try ensureLogSignaturesForProgram(&semantics_result.symbols, program.nodes);

    var type_resolver = lib.TypeResolver.init(allocator, program.arena.allocator(), &semantics_result.symbols);
    defer type_resolver.deinit();
    try type_resolver.resolveTypes(program.nodes);

    return program;
}

pub fn loadProgramWithImportsTyped(allocator: std.mem.Allocator, entry_file_path: []const u8) !ParsedProgram {
    return loadProgramWithImportsTypedWithResolverOptions(allocator, entry_file_path, .{});
}
