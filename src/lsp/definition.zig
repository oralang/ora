const std = @import("std");
const lexer = @import("ora_lexer");
const parser = @import("../parser/mod.zig");
const ast = @import("ora_ast");
const frontend = @import("frontend.zig");

const Allocator = std.mem.Allocator;
const ResolveError = Allocator.Error;

pub const Definition = struct {
    uri: ?[]const u8 = null,
    range: frontend.Range,
};

pub const ImportBinding = struct {
    alias: []const u8,
    target_uri: []const u8,
    target_source: ?[]const u8,
};

pub const CrossFileContext = struct {
    bindings: []const ImportBinding,
};

pub fn definitionAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Definition {
    return definitionAtImpl(allocator, source, position, null);
}

pub fn definitionAtCrossFile(allocator: Allocator, source: []const u8, position: frontend.Position, cross_file: CrossFileContext) !?Definition {
    return definitionAtImpl(allocator, source, position, cross_file);
}

fn definitionAtImpl(allocator: Allocator, source: []const u8, position: frontend.Position, cross_file: ?CrossFileContext) !?Definition {
    var lex = try lexer.Lexer.initWithConfig(allocator, source, lexer.LexerConfig.development());
    defer lex.deinit();

    const tokens = lex.scanTokens() catch return null;
    defer allocator.free(tokens);

    const previous_parser_stderr = parser.diagnostics.enable_stderr_diagnostics;
    parser.diagnostics.enable_stderr_diagnostics = false;
    defer parser.diagnostics.enable_stderr_diagnostics = previous_parser_stderr;

    var parse_result = parser.parseRaw(allocator, tokens) catch return null;
    defer parse_result.arena.deinit();

    var ctx = Context{
        .allocator = allocator,
        .query_position = position,
        .cross_file = cross_file,
    };
    defer ctx.deinit();

    try collectDeclarations(&ctx, parse_result.nodes);

    for (parse_result.nodes) |node| {
        if (try resolveNode(&ctx, node, null)) {
            return .{ .uri = ctx.found_uri, .range = ctx.found_range.? };
        }
    }

    return null;
}

fn findImportBinding(cross_file: ?CrossFileContext, name: []const u8) ?ImportBinding {
    const cf = cross_file orelse return null;
    for (cf.bindings) |binding| {
        if (std.mem.eql(u8, binding.alias, name)) return binding;
    }
    return null;
}

fn findTopLevelDeclarationInSource(allocator: Allocator, source: []const u8, name: []const u8) !?frontend.Range {
    var lex = try lexer.Lexer.initWithConfig(allocator, source, lexer.LexerConfig.development());
    defer lex.deinit();

    const tokens = lex.scanTokens() catch return null;
    defer allocator.free(tokens);

    const prev_stderr = parser.diagnostics.enable_stderr_diagnostics;
    parser.diagnostics.enable_stderr_diagnostics = false;
    defer parser.diagnostics.enable_stderr_diagnostics = prev_stderr;

    var result = parser.parseRaw(allocator, tokens) catch return null;
    defer result.arena.deinit();

    for (result.nodes) |node| {
        const decl_name: ?[]const u8 = switch (node) {
            .Contract => |d| d.name,
            .Function => |d| d.name,
            .VariableDecl => |d| d.name,
            .Constant => |d| d.name,
            .StructDecl => |d| d.name,
            .BitfieldDecl => |d| d.name,
            .EnumDecl => |d| d.name,
            .LogDecl => |d| d.name,
            .ErrorDecl => |d| d.name,
            else => null,
        };
        if (decl_name) |dn| {
            if (std.mem.eql(u8, dn, name)) {
                const span = switch (node) {
                    .Contract => |d| d.span,
                    .Function => |d| d.span,
                    .VariableDecl => |d| d.span,
                    .Constant => |d| d.span,
                    .StructDecl => |d| d.span,
                    .BitfieldDecl => |d| d.span,
                    .EnumDecl => |d| d.span,
                    .LogDecl => |d| d.span,
                    .ErrorDecl => |d| d.span,
                    else => unreachable,
                };
                return selectionRange(span, dn);
            }
        }
    }
    return null;
}

const file_origin: frontend.Range = .{
    .start = .{ .line = 0, .character = 0 },
    .end = .{ .line = 0, .character = 0 },
};

const Declaration = struct {
    name: []const u8,
    range: frontend.Range,
};

const ContractMemberDeclaration = struct {
    contract_name: []const u8,
    declaration: Declaration,
};

const LocalBinding = struct {
    name: []const u8,
    range: frontend.Range,
    depth: usize,
};

const Context = struct {
    allocator: Allocator,
    query_position: frontend.Position,
    scope_depth: usize = 0,
    found_range: ?frontend.Range = null,
    found_uri: ?[]const u8 = null,
    cross_file: ?CrossFileContext = null,
    global_declarations: std.ArrayList(Declaration) = .{},
    contract_member_declarations: std.ArrayList(ContractMemberDeclaration) = .{},
    local_bindings: std.ArrayList(LocalBinding) = .{},

    fn deinit(self: *Context) void {
        self.global_declarations.deinit(self.allocator);
        self.contract_member_declarations.deinit(self.allocator);
        self.local_bindings.deinit(self.allocator);
    }

    fn pushScope(self: *Context) void {
        self.scope_depth += 1;
    }

    fn popScope(self: *Context) void {
        if (self.scope_depth == 0) return;
        self.scope_depth -= 1;
    }

    fn bindLocal(self: *Context, name: []const u8, span: ast.SourceSpan) !void {
        try self.local_bindings.append(self.allocator, .{
            .name = name,
            .range = selectionRange(span, name),
            .depth = self.scope_depth,
        });
    }

    fn bindTopLevel(self: *Context, name: []const u8, span: ast.SourceSpan) !void {
        try self.global_declarations.append(self.allocator, .{
            .name = name,
            .range = selectionRange(span, name),
        });
    }

    fn bindContractMember(self: *Context, contract_name: []const u8, name: []const u8, span: ast.SourceSpan) !void {
        try self.contract_member_declarations.append(self.allocator, .{
            .contract_name = contract_name,
            .declaration = .{
                .name = name,
                .range = selectionRange(span, name),
            },
        });
    }
};

fn collectDeclarations(ctx: *Context, nodes: []const ast.AstNode) !void {
    for (nodes) |node| {
        switch (node) {
            .Contract => |contract_decl| {
                try ctx.bindTopLevel(contract_decl.name, contract_decl.span);
                for (contract_decl.body) |member| {
                    try collectContractMemberDeclaration(ctx, contract_decl.name, member);
                }
            },
            .Function => |function_decl| try ctx.bindTopLevel(function_decl.name, function_decl.span),
            .VariableDecl => |variable_decl| try ctx.bindTopLevel(variable_decl.name, variable_decl.span),
            .Constant => |constant_decl| try ctx.bindTopLevel(constant_decl.name, constant_decl.span),
            .StructDecl => |struct_decl| try ctx.bindTopLevel(struct_decl.name, struct_decl.span),
            .BitfieldDecl => |bitfield_decl| try ctx.bindTopLevel(bitfield_decl.name, bitfield_decl.span),
            .EnumDecl => |enum_decl| try ctx.bindTopLevel(enum_decl.name, enum_decl.span),
            .LogDecl => |log_decl| try ctx.bindTopLevel(log_decl.name, log_decl.span),
            .ErrorDecl => |error_decl| try ctx.bindTopLevel(error_decl.name, error_decl.span),
            .Import => |import_decl| {
                if (import_decl.alias) |alias| {
                    try ctx.bindTopLevel(alias, import_decl.span);
                }
            },
            else => {},
        }
    }
}

fn collectContractMemberDeclaration(ctx: *Context, contract_name: []const u8, member: ast.AstNode) !void {
    switch (member) {
        .Function => |function_decl| try ctx.bindContractMember(contract_name, function_decl.name, function_decl.span),
        .VariableDecl => |variable_decl| try ctx.bindContractMember(contract_name, variable_decl.name, variable_decl.span),
        .Constant => |constant_decl| try ctx.bindContractMember(contract_name, constant_decl.name, constant_decl.span),
        .StructDecl => |struct_decl| try ctx.bindContractMember(contract_name, struct_decl.name, struct_decl.span),
        .BitfieldDecl => |bitfield_decl| try ctx.bindContractMember(contract_name, bitfield_decl.name, bitfield_decl.span),
        .EnumDecl => |enum_decl| try ctx.bindContractMember(contract_name, enum_decl.name, enum_decl.span),
        .LogDecl => |log_decl| try ctx.bindContractMember(contract_name, log_decl.name, log_decl.span),
        .ErrorDecl => |error_decl| try ctx.bindContractMember(contract_name, error_decl.name, error_decl.span),
        .Import => |import_decl| {
            if (import_decl.alias) |alias| {
                try ctx.bindContractMember(contract_name, alias, import_decl.span);
            }
        },
        else => {},
    }
}

fn resolveNode(ctx: *Context, node: ast.AstNode, active_contract: ?[]const u8) ResolveError!bool {
    if (ctx.found_range != null) return true;

    switch (node) {
        .Contract => |contract_decl| {
            if (matchDeclarationAtQuery(ctx, contract_decl.name, contract_decl.span)) return true;
            for (contract_decl.body) |member| {
                if (try resolveNode(ctx, member, contract_decl.name)) return true;
            }
        },
        .Function => |function_decl| {
            if (try resolveFunction(ctx, function_decl, active_contract)) return true;
        },
        .VariableDecl => |variable_decl| {
            if (matchDeclarationAtQuery(ctx, variable_decl.name, variable_decl.span)) return true;
            if (variable_decl.value) |value| {
                if (try resolveExpression(ctx, value, active_contract)) return true;
            }
        },
        .Constant => |constant_decl| {
            if (matchDeclarationAtQuery(ctx, constant_decl.name, constant_decl.span)) return true;
            if (try resolveExpression(ctx, constant_decl.value, active_contract)) return true;
        },
        .StructDecl => |struct_decl| {
            if (matchDeclarationAtQuery(ctx, struct_decl.name, struct_decl.span)) return true;
            for (struct_decl.fields) |field| {
                if (matchDeclarationAtQuery(ctx, field.name, field.span)) return true;
            }
        },
        .BitfieldDecl => |bitfield_decl| {
            if (matchDeclarationAtQuery(ctx, bitfield_decl.name, bitfield_decl.span)) return true;
            for (bitfield_decl.fields) |field| {
                if (matchDeclarationAtQuery(ctx, field.name, field.span)) return true;
            }
        },
        .EnumDecl => |enum_decl| {
            if (matchDeclarationAtQuery(ctx, enum_decl.name, enum_decl.span)) return true;
            for (enum_decl.variants) |variant| {
                if (matchDeclarationAtQuery(ctx, variant.name, variant.span)) return true;
                if (variant.value) |*value| {
                    if (try resolveExpression(ctx, value, active_contract)) return true;
                }
            }
        },
        .LogDecl => |log_decl| {
            if (matchDeclarationAtQuery(ctx, log_decl.name, log_decl.span)) return true;
            for (log_decl.fields) |field| {
                if (matchDeclarationAtQuery(ctx, field.name, field.span)) return true;
            }
        },
        .ErrorDecl => |error_decl| {
            if (matchDeclarationAtQuery(ctx, error_decl.name, error_decl.span)) return true;
            if (error_decl.parameters) |parameters| {
                for (parameters) |parameter| {
                    if (matchDeclarationAtQuery(ctx, parameter.name, parameter.span)) return true;
                }
            }
        },
        .Import => |import_decl| {
            if (rangeContainsPosition(spanToRange(import_decl.span), ctx.query_position)) {
                if (findImportBinding(ctx.cross_file, import_decl.alias orelse import_decl.path)) |binding| {
                    ctx.found_uri = binding.target_uri;
                    ctx.found_range = file_origin;
                    return true;
                }
            }
            if (import_decl.alias) |alias| {
                if (matchDeclarationAtQuery(ctx, alias, import_decl.span)) return true;
            }
        },
        .Block => |block| if (try resolveBlock(ctx, block, active_contract)) return true,
        .Statement => |stmt| if (try resolveStatement(ctx, stmt.*, active_contract)) return true,
        .Expression => |expr| if (try resolveExpression(ctx, expr, active_contract)) return true,
        .TryBlock => |try_block| {
            if (try resolveBlock(ctx, try_block.try_block, active_contract)) return true;
            if (try_block.catch_block) |catch_block| {
                if (try resolveBlock(ctx, catch_block.block, active_contract)) return true;
            }
        },
        else => {},
    }

    return false;
}

fn resolveFunction(ctx: *Context, function_decl: ast.FunctionNode, active_contract: ?[]const u8) ResolveError!bool {
    if (matchDeclarationAtQuery(ctx, function_decl.name, function_decl.span)) return true;

    ctx.pushScope();
    defer ctx.popScope();

    for (function_decl.parameters) |parameter| {
        if (matchDeclarationAtQuery(ctx, parameter.name, parameter.span)) return true;
        try ctx.bindLocal(parameter.name, parameter.span);
        if (parameter.default_value) |default_value| {
            if (try resolveExpression(ctx, default_value, active_contract)) return true;
        }
    }

    for (function_decl.requires_clauses) |clause| {
        if (try resolveExpression(ctx, clause, active_contract)) return true;
    }
    for (function_decl.ensures_clauses) |clause| {
        if (try resolveExpression(ctx, clause, active_contract)) return true;
    }

    return resolveBlock(ctx, function_decl.body, active_contract);
}

fn resolveBlock(ctx: *Context, block: ast.Statements.BlockNode, active_contract: ?[]const u8) ResolveError!bool {
    ctx.pushScope();
    defer ctx.popScope();

    for (block.statements) |stmt| {
        if (try resolveStatement(ctx, stmt, active_contract)) return true;
    }
    return false;
}

fn resolveStatement(ctx: *Context, stmt: ast.Statements.StmtNode, active_contract: ?[]const u8) ResolveError!bool {
    switch (stmt) {
        .Expr => |expr| return resolveExpression(ctx, &expr, active_contract),
        .VariableDecl => |variable_decl| {
            if (variable_decl.value) |value| {
                if (try resolveExpression(ctx, value, active_contract)) return true;
            }

            if (matchDeclarationAtQuery(ctx, variable_decl.name, variable_decl.span)) return true;
            try ctx.bindLocal(variable_decl.name, variable_decl.span);

            if (variable_decl.tuple_names) |names| {
                for (names) |name| {
                    try ctx.bindLocal(name, variable_decl.span);
                }
            }
            return false;
        },
        .DestructuringAssignment => |destructure| return resolveExpression(ctx, destructure.value, active_contract),
        .Return => |ret| {
            if (ret.value) |*value| {
                return resolveExpression(ctx, value, active_contract);
            }
            return false;
        },
        .If => |if_stmt| {
            if (try resolveExpression(ctx, &if_stmt.condition, active_contract)) return true;
            if (try resolveBlock(ctx, if_stmt.then_branch, active_contract)) return true;
            if (if_stmt.else_branch) |else_branch| {
                if (try resolveBlock(ctx, else_branch, active_contract)) return true;
            }
            return false;
        },
        .While => |while_stmt| {
            if (try resolveExpression(ctx, &while_stmt.condition, active_contract)) return true;
            for (while_stmt.invariants) |*invariant| {
                if (try resolveExpression(ctx, invariant, active_contract)) return true;
            }
            if (while_stmt.decreases) |decreases| {
                if (try resolveExpression(ctx, decreases, active_contract)) return true;
            }
            if (while_stmt.increases) |increases| {
                if (try resolveExpression(ctx, increases, active_contract)) return true;
            }
            return resolveBlock(ctx, while_stmt.body, active_contract);
        },
        .ForLoop => |for_loop| {
            if (try resolveExpression(ctx, &for_loop.iterable, active_contract)) return true;
            for (for_loop.invariants) |*invariant| {
                if (try resolveExpression(ctx, invariant, active_contract)) return true;
            }
            if (for_loop.decreases) |decreases| {
                if (try resolveExpression(ctx, decreases, active_contract)) return true;
            }
            if (for_loop.increases) |increases| {
                if (try resolveExpression(ctx, increases, active_contract)) return true;
            }

            ctx.pushScope();
            defer ctx.popScope();

            switch (for_loop.pattern) {
                .Single => |single| {
                    if (matchDeclarationAtQuery(ctx, single.name, single.span)) return true;
                    try ctx.bindLocal(single.name, single.span);
                },
                .IndexPair => |pair| {
                    if (matchDeclarationAtQuery(ctx, pair.item, pair.span)) return true;
                    if (matchDeclarationAtQuery(ctx, pair.index, pair.span)) return true;
                    try ctx.bindLocal(pair.item, pair.span);
                    try ctx.bindLocal(pair.index, pair.span);
                },
                .Destructured => |destructured| {
                    switch (destructured.pattern) {
                        .Struct => |fields| {
                            for (fields) |field| {
                                if (matchDeclarationAtQuery(ctx, field.variable, field.span)) return true;
                                try ctx.bindLocal(field.variable, field.span);
                            }
                        },
                        .Tuple => |names| {
                            for (names) |name| {
                                if (matchDeclarationAtQuery(ctx, name, destructured.span)) return true;
                                try ctx.bindLocal(name, destructured.span);
                            }
                        },
                        .Array => |names| {
                            for (names) |name| {
                                if (matchDeclarationAtQuery(ctx, name, destructured.span)) return true;
                                try ctx.bindLocal(name, destructured.span);
                            }
                        },
                    }
                },
            }

            return resolveBlock(ctx, for_loop.body, active_contract);
        },
        .Break => |break_stmt| {
            if (break_stmt.value) |value| {
                return resolveExpression(ctx, value, active_contract);
            }
            return false;
        },
        .Continue => |continue_stmt| {
            if (continue_stmt.value) |value| {
                return resolveExpression(ctx, value, active_contract);
            }
            return false;
        },
        .Log => |log_stmt| {
            for (log_stmt.args) |*arg| {
                if (try resolveExpression(ctx, arg, active_contract)) return true;
            }
            return false;
        },
        .Lock => |lock_stmt| return resolveExpression(ctx, &lock_stmt.path, active_contract),
        .Unlock => |unlock_stmt| return resolveExpression(ctx, &unlock_stmt.path, active_contract),
        .Assert => |assert_stmt| return resolveExpression(ctx, &assert_stmt.condition, active_contract),
        .Invariant => |invariant_stmt| return resolveExpression(ctx, &invariant_stmt.condition, active_contract),
        .Requires => |requires_stmt| return resolveExpression(ctx, &requires_stmt.condition, active_contract),
        .Ensures => |ensures_stmt| return resolveExpression(ctx, &ensures_stmt.condition, active_contract),
        .Assume => |assume_stmt| return resolveExpression(ctx, &assume_stmt.condition, active_contract),
        .Havoc => |havoc_stmt| {
            _ = havoc_stmt;
            return false;
        },
        .Switch => |switch_stmt| {
            if (try resolveExpression(ctx, &switch_stmt.condition, active_contract)) return true;
            for (switch_stmt.cases) |switch_case| {
                if (try resolveSwitchPattern(ctx, switch_case.pattern, active_contract)) return true;
                if (try resolveSwitchBody(ctx, switch_case.body, active_contract)) return true;
            }
            if (switch_stmt.default_case) |default_case| {
                if (try resolveBlock(ctx, default_case, active_contract)) return true;
            }
            return false;
        },
        .TryBlock => |try_block| {
            if (try resolveBlock(ctx, try_block.try_block, active_contract)) return true;
            if (try_block.catch_block) |catch_block| {
                ctx.pushScope();
                defer ctx.popScope();

                if (catch_block.error_variable) |error_name| {
                    if (matchDeclarationAtQuery(ctx, error_name, catch_block.span)) return true;
                    try ctx.bindLocal(error_name, catch_block.span);
                }
                return resolveBlock(ctx, catch_block.block, active_contract);
            }
            return false;
        },
        .ErrorDecl => |error_decl| {
            if (matchDeclarationAtQuery(ctx, error_decl.name, error_decl.span)) return true;
            if (error_decl.parameters) |parameters| {
                for (parameters) |parameter| {
                    if (matchDeclarationAtQuery(ctx, parameter.name, parameter.span)) return true;
                }
            }
            return false;
        },
        .CompoundAssignment => |compound| {
            if (try resolveExpression(ctx, compound.target, active_contract)) return true;
            return resolveExpression(ctx, compound.value, active_contract);
        },
        .LabeledBlock => |labeled| return resolveBlock(ctx, labeled.block, active_contract),
    }
}

fn resolveExpression(ctx: *Context, expr: *const ast.Expressions.ExprNode, active_contract: ?[]const u8) ResolveError!bool {
    switch (expr.*) {
        .Identifier => |identifier| {
            if (rangeContainsPosition(spanToRange(identifier.span), ctx.query_position)) {
                if (resolveName(ctx, identifier.name, active_contract)) |decl| {
                    if (findImportBinding(ctx.cross_file, identifier.name)) |binding| {
                        ctx.found_uri = binding.target_uri;
                        ctx.found_range = file_origin;
                        return true;
                    }
                    ctx.found_range = decl.range;
                    return true;
                }
            }
            return false;
        },
        .Literal => return false,
        .Binary => |binary| {
            if (try resolveExpression(ctx, binary.lhs, active_contract)) return true;
            return resolveExpression(ctx, binary.rhs, active_contract);
        },
        .Unary => |unary| return resolveExpression(ctx, unary.operand, active_contract),
        .Assignment => |assignment| {
            if (try resolveExpression(ctx, assignment.target, active_contract)) return true;
            return resolveExpression(ctx, assignment.value, active_contract);
        },
        .CompoundAssignment => |compound| {
            if (try resolveExpression(ctx, compound.target, active_contract)) return true;
            return resolveExpression(ctx, compound.value, active_contract);
        },
        .Call => |call| {
            if (try resolveExpression(ctx, call.callee, active_contract)) return true;
            for (call.arguments) |argument| {
                if (try resolveExpression(ctx, argument, active_contract)) return true;
            }
            return false;
        },
        .Index => |index| {
            if (try resolveExpression(ctx, index.target, active_contract)) return true;
            return resolveExpression(ctx, index.index, active_contract);
        },
        .FieldAccess => |field_access| {
            if (ctx.cross_file != null) {
                if (field_access.target.* == .Identifier) {
                    const target_name = field_access.target.Identifier.name;
                    if (findImportBinding(ctx.cross_file, target_name)) |binding| {
                        if (rangeContainsPosition(spanToRange(field_access.span), ctx.query_position)) {
                            if (binding.target_source) |target_source| {
                                if (findTopLevelDeclarationInSource(ctx.allocator, target_source, field_access.field) catch null) |member_range| {
                                    ctx.found_uri = binding.target_uri;
                                    ctx.found_range = member_range;
                                    return true;
                                }
                            }
                            ctx.found_uri = binding.target_uri;
                            ctx.found_range = file_origin;
                            return true;
                        }
                    }
                }
            }
            return resolveExpression(ctx, field_access.target, active_contract);
        },
        .Cast => |cast| return resolveExpression(ctx, cast.operand, active_contract),
        .Comptime => |comptime_expr| return resolveBlock(ctx, comptime_expr.block, active_contract),
        .Old => |old_expr| return resolveExpression(ctx, old_expr.expr, active_contract),
        .Quantified => |quantified| {
            ctx.pushScope();
            defer ctx.popScope();

            try ctx.bindLocal(quantified.variable, quantified.span);
            if (quantified.condition) |condition| {
                if (try resolveExpression(ctx, condition, active_contract)) return true;
            }
            return resolveExpression(ctx, quantified.body, active_contract);
        },
        .Tuple => |tuple| {
            for (tuple.elements) |element| {
                if (try resolveExpression(ctx, element, active_contract)) return true;
            }
            return false;
        },
        .SwitchExpression => |switch_expr| {
            if (try resolveExpression(ctx, switch_expr.condition, active_contract)) return true;
            for (switch_expr.cases) |switch_case| {
                if (try resolveSwitchPattern(ctx, switch_case.pattern, active_contract)) return true;
                if (try resolveSwitchBody(ctx, switch_case.body, active_contract)) return true;
            }
            if (switch_expr.default_case) |default_case| {
                if (try resolveBlock(ctx, default_case, active_contract)) return true;
            }
            return false;
        },
        .Try => |try_expr| return resolveExpression(ctx, try_expr.expr, active_contract),
        .ErrorReturn => |error_return| {
            if (rangeContainsPosition(spanToRange(error_return.span), ctx.query_position)) {
                if (resolveName(ctx, error_return.error_name, active_contract)) |decl| {
                    ctx.found_range = decl.range;
                    return true;
                }
            }
            if (error_return.parameters) |parameters| {
                for (parameters) |parameter| {
                    if (try resolveExpression(ctx, parameter, active_contract)) return true;
                }
            }
            return false;
        },
        .ErrorCast => |error_cast| return resolveExpression(ctx, error_cast.operand, active_contract),
        .Shift => |shift| {
            if (try resolveExpression(ctx, shift.mapping, active_contract)) return true;
            if (try resolveExpression(ctx, shift.source, active_contract)) return true;
            if (try resolveExpression(ctx, shift.dest, active_contract)) return true;
            return resolveExpression(ctx, shift.amount, active_contract);
        },
        .StructInstantiation => |struct_instantiation| {
            if (try resolveExpression(ctx, struct_instantiation.struct_name, active_contract)) return true;
            for (struct_instantiation.fields) |field| {
                if (try resolveExpression(ctx, field.value, active_contract)) return true;
            }
            return false;
        },
        .AnonymousStruct => |anonymous_struct| {
            for (anonymous_struct.fields) |field| {
                if (try resolveExpression(ctx, field.value, active_contract)) return true;
            }
            return false;
        },
        .Range => |range_expr| {
            if (try resolveExpression(ctx, range_expr.start, active_contract)) return true;
            return resolveExpression(ctx, range_expr.end, active_contract);
        },
        .LabeledBlock => |labeled| return resolveBlock(ctx, labeled.block, active_contract),
        .Destructuring => |destructuring| return resolveExpression(ctx, destructuring.value, active_contract),
        .EnumLiteral => |enum_literal| {
            if (rangeContainsPosition(spanToRange(enum_literal.span), ctx.query_position)) {
                if (resolveName(ctx, enum_literal.enum_name, active_contract)) |decl| {
                    ctx.found_range = decl.range;
                    return true;
                }
                if (resolveName(ctx, enum_literal.variant_name, active_contract)) |decl| {
                    ctx.found_range = decl.range;
                    return true;
                }
            }
            return false;
        },
        .ArrayLiteral => |array_literal| {
            for (array_literal.elements) |element| {
                if (try resolveExpression(ctx, element, active_contract)) return true;
            }
            return false;
        },
    }
}

fn resolveSwitchPattern(ctx: *Context, pattern: ast.Switch.Pattern, active_contract: ?[]const u8) ResolveError!bool {
    switch (pattern) {
        .Range => |range_expr| {
            if (try resolveExpression(ctx, range_expr.start, active_contract)) return true;
            return resolveExpression(ctx, range_expr.end, active_contract);
        },
        else => return false,
    }
}

fn resolveSwitchBody(ctx: *Context, body: ast.Switch.Body, active_contract: ?[]const u8) ResolveError!bool {
    switch (body) {
        .Expression => |expr| return resolveExpression(ctx, expr, active_contract),
        .Block => |block| return resolveBlock(ctx, block, active_contract),
        .LabeledBlock => |labeled| return resolveBlock(ctx, labeled.block, active_contract),
    }
}

fn resolveName(ctx: *Context, name: []const u8, active_contract: ?[]const u8) ?Declaration {
    var binding_index = ctx.local_bindings.items.len;
    while (binding_index > 0) {
        binding_index -= 1;
        const binding = ctx.local_bindings.items[binding_index];
        if (binding.depth > ctx.scope_depth) continue;
        if (std.mem.eql(u8, binding.name, name)) {
            return .{ .name = binding.name, .range = binding.range };
        }
    }

    if (active_contract) |contract_name| {
        var member_index = ctx.contract_member_declarations.items.len;
        while (member_index > 0) {
            member_index -= 1;
            const member = ctx.contract_member_declarations.items[member_index];
            if (!std.mem.eql(u8, member.contract_name, contract_name)) continue;
            if (std.mem.eql(u8, member.declaration.name, name)) {
                return member.declaration;
            }
        }
    }

    var global_index = ctx.global_declarations.items.len;
    while (global_index > 0) {
        global_index -= 1;
        const declaration = ctx.global_declarations.items[global_index];
        if (std.mem.eql(u8, declaration.name, name)) {
            return declaration;
        }
    }

    return null;
}

fn matchDeclarationAtQuery(ctx: *Context, name: []const u8, span: ast.SourceSpan) bool {
    const range = selectionRange(span, name);
    if (!rangeContainsPosition(range, ctx.query_position)) return false;
    ctx.found_range = range;
    return true;
}

fn spanToRange(span: ast.SourceSpan) frontend.Range {
    const start_line = if (span.line > 0) span.line - 1 else 0;
    const start_character = if (span.column > 0) span.column - 1 else 0;
    const end_character = std.math.add(u32, start_character, span.length) catch std.math.maxInt(u32);

    return .{
        .start = .{ .line = start_line, .character = start_character },
        .end = .{ .line = start_line, .character = end_character },
    };
}

fn selectionRange(span: ast.SourceSpan, name: []const u8) frontend.Range {
    var range = spanToRange(span);
    const name_len = std.math.cast(u32, name.len) orelse std.math.maxInt(u32);
    range.end.character = std.math.add(u32, range.start.character, name_len) catch std.math.maxInt(u32);
    return range;
}

fn rangeContainsPosition(range: frontend.Range, position: frontend.Position) bool {
    if (positionLessThan(position, range.start)) return false;
    if (!positionLessThan(position, range.end)) return false;
    return true;
}

fn positionLessThan(lhs: frontend.Position, rhs: frontend.Position) bool {
    if (lhs.line < rhs.line) return true;
    if (lhs.line > rhs.line) return false;
    return lhs.character < rhs.character;
}
