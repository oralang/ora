const std = @import("std");
const ast = @import("../ast/mod.zig");
const diagnostics = @import("../diagnostics/mod.zig");
const model = @import("model.zig");
const source = @import("../source/mod.zig");

const ItemIndexResult = model.ItemIndexResult;
const NameResolutionResult = model.NameResolutionResult;
const ResolvedBinding = model.ResolvedBinding;

pub fn resolveNames(
    allocator: std.mem.Allocator,
    file_id: source.FileId,
    file: *const ast.AstFile,
    item_index: *const ItemIndexResult,
) !NameResolutionResult {
    var result = NameResolutionResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .expr_bindings = &[_]?ResolvedBinding{},
        .diagnostics = diagnostics.DiagnosticList.init(allocator),
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    const bindings = try arena.alloc(?ResolvedBinding, file.expressions.len);
    @memset(bindings, null);

    var root_env = try Env.init(arena, null);
    for (item_index.entries) |entry| {
        if (std.mem.indexOfScalar(u8, entry.name, '.')) |_| continue;
        try root_env.bindings.put(entry.name, .{ .item = entry.item_id });
    }
    for (file.root_items) |item_id| {
        const item = file.item(item_id).*;
        if (item != .Import) continue;
        if (item.Import.alias) |alias| {
            try root_env.bindings.put(alias, .{ .item = item_id });
        }
    }

    var resolver = Resolver{
        .arena = arena,
        .file_id = file_id,
        .file = file,
        .item_index = item_index,
        .bindings = bindings,
        .diagnostics = &result.diagnostics,
    };

    for (file.root_items) |item_id| {
        try resolver.resolveItem(item_id, &root_env);
    }

    result.expr_bindings = bindings;
    return result;
}

const Env = struct {
    parent: ?*const Env,
    bindings: std.StringHashMap(ResolvedBinding),

    fn init(allocator: std.mem.Allocator, parent: ?*const Env) !Env {
        return .{
            .parent = parent,
            .bindings = std.StringHashMap(ResolvedBinding).init(allocator),
        };
    }

    fn lookup(self: *const Env, name: []const u8) ?ResolvedBinding {
        if (self.bindings.get(name)) |binding| return binding;
        if (self.parent) |parent| return parent.lookup(name);
        return null;
    }
};

const Resolver = struct {
    arena: std.mem.Allocator,
    file_id: source.FileId,
    file: *const ast.AstFile,
    item_index: *const ItemIndexResult,
    bindings: []?ResolvedBinding,
    diagnostics: *diagnostics.DiagnosticList,

    fn resolveItem(self: *Resolver, item_id: ast.ItemId, env: *const Env) anyerror!void {
        switch (self.file.item(item_id).*) {
            .Contract => |contract| {
                var contract_env = try Env.init(self.arena, env);
                for (contract.members) |member_id| {
                    const member = self.file.item(member_id).*;
                    const name = switch (member) {
                        .Function => member.Function.name,
                        .Field => member.Field.name,
                        .Constant => member.Constant.name,
                        .Struct => member.Struct.name,
                        .Bitfield => member.Bitfield.name,
                        .Enum => member.Enum.name,
                        .TypeAlias => member.TypeAlias.name,
                        .LogDecl => member.LogDecl.name,
                        .ErrorDecl => member.ErrorDecl.name,
                        .GhostBlock => continue,
                        else => continue,
                    };
                    try contract_env.bindings.put(name, .{ .item = member_id });
                }
                for (contract.invariants) |expr_id| {
                    try self.resolveExpr(expr_id, &contract_env);
                }
                for (contract.members) |member_id| {
                    try self.resolveItem(member_id, &contract_env);
                }
            },
            .Function => |function| {
                var function_env = try Env.init(self.arena, env);
                for (function.parameters) |parameter| {
                    try self.bindPatternIfName(&function_env, parameter.pattern);
                }
                for (function.clauses) |clause| {
                    try self.resolveExpr(clause.expr, &function_env);
                }
                try self.resolveBody(function.body, &function_env);
            },
            .Trait => |trait_item| {
                var trait_env = try Env.init(self.arena, env);
                for (trait_item.methods) |method| {
                    var method_env = try Env.init(self.arena, &trait_env);
                    for (method.parameters) |parameter| {
                        try self.bindPatternIfName(&method_env, parameter.pattern);
                    }
                    for (method.clauses) |clause| {
                        try self.resolveExpr(clause.expr, &method_env);
                    }
                }
            },
            .Impl => |impl_item| {
                var impl_env = try Env.init(self.arena, env);
                for (impl_item.methods) |method_id| {
                    const item = self.file.item(method_id).*;
                    if (item != .Function) continue;
                    try impl_env.bindings.put(item.Function.name, .{ .item = method_id });
                }
                if (self.firstImplSelfPattern(impl_item)) |pattern_id| {
                    const pattern = self.file.pattern(pattern_id).*;
                    if (pattern == .Name) {
                        try impl_env.bindings.put(pattern.Name.name, .{ .pattern = pattern_id });
                    }
                }
                if (self.lookupTraitGhostItem(impl_item.trait_name, env)) |ghost_id| {
                    try self.resolveItem(ghost_id, &impl_env);
                }
                for (impl_item.methods) |method_id| {
                    try self.resolveItem(method_id, &impl_env);
                }
            },
            .Enum => |enum_item| {
                var enum_env = try Env.init(self.arena, env);
                for (enum_item.variants) |variant| {
                    try enum_env.bindings.put(variant.name, .{ .item = item_id });
                }
                for (enum_item.variants) |variant| {
                    if (variant.value) |expr_id| try self.resolveExpr(expr_id, &enum_env);
                }
            },
            .Field => |field| {
                if (field.value) |expr_id| try self.resolveExpr(expr_id, env);
            },
            .Constant => |constant| try self.resolveExpr(constant.value, env),
            .GhostBlock => |ghost_block| try self.resolveBody(ghost_block.body, env),
            else => {},
        }
    }

    fn resolveBody(self: *Resolver, body_id: ast.BodyId, env: *const Env) anyerror!void {
        var body_env = try Env.init(self.arena, env);
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            try self.resolveStmt(statement_id, &body_env);
        }
    }

    fn resolveSwitchPattern(self: *Resolver, pattern: ast.SwitchPattern, env: *const Env, arm_env: *Env) anyerror!void {
        switch (pattern) {
            .Expr => |expr_id| try self.resolveExpr(expr_id, env),
            .Range => |range_pattern| {
                try self.resolveExpr(range_pattern.start, env);
                try self.resolveExpr(range_pattern.end, env);
            },
            .Or => |or_pattern| {
                for (or_pattern.alternatives) |alternative| {
                    try self.resolveSwitchPattern(alternative, env, arm_env);
                }
            },
            .Ok, .Err => |pattern_id| try self.bindPatternIfName(arm_env, pattern_id),
            .NamedError => |named_error| {
                try self.resolveExpr(named_error.callee, env);
                for (named_error.bindings) |pattern_id| try self.bindPatternIfName(arm_env, pattern_id);
            },
            .Else => {},
        }
    }

    fn resolveStmt(self: *Resolver, statement_id: ast.StmtId, env: *Env) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| {
                if (decl.value) |expr_id| try self.resolveExpr(expr_id, env);
                try self.bindPatternIfName(env, decl.pattern);
            },
            .Return => |ret| if (ret.value) |expr_id| try self.resolveExpr(expr_id, env),
            .If => |if_stmt| {
                try self.resolveExpr(if_stmt.condition, env);
                try self.resolveBody(if_stmt.then_body, env);
                if (if_stmt.else_body) |else_body| try self.resolveBody(else_body, env);
            },
            .While => |while_stmt| {
                try self.resolveExpr(while_stmt.condition, env);
                for (while_stmt.invariants) |expr_id| try self.resolveExpr(expr_id, env);
                try self.resolveBody(while_stmt.body, env);
            },
            .For => |for_stmt| {
                try self.resolveExpr(for_stmt.iterable, env);
                var loop_env = try Env.init(self.arena, env);
                try self.bindPatternIfName(&loop_env, for_stmt.item_pattern);
                if (for_stmt.index_pattern) |index_pattern| try self.bindPatternIfName(&loop_env, index_pattern);
                for (for_stmt.invariants) |expr_id| try self.resolveExpr(expr_id, &loop_env);
                try self.resolveBody(for_stmt.body, &loop_env);
            },
            .Switch => |switch_stmt| {
                try self.resolveExpr(switch_stmt.condition, env);
                for (switch_stmt.arms) |arm| {
                    var arm_env = try Env.init(self.arena, env);
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.resolveExpr(expr_id, env),
                        .Range => |range_pattern| {
                            try self.resolveExpr(range_pattern.start, env);
                            try self.resolveExpr(range_pattern.end, env);
                        },
                        .Or => |or_pattern| {
                            for (or_pattern.alternatives) |alternative| {
                                try self.resolveSwitchPattern(alternative, env, &arm_env);
                            }
                        },
                        .Ok, .Err => |pattern_id| try self.bindPatternIfName(&arm_env, pattern_id),
                        .NamedError => |named_error| {
                            try self.resolveExpr(named_error.callee, env);
                            for (named_error.bindings) |pattern_id| try self.bindPatternIfName(&arm_env, pattern_id);
                        },
                        .Else => {},
                    }
                    try self.resolveBody(arm.body, &arm_env);
                }
                if (switch_stmt.else_body) |else_body| try self.resolveBody(else_body, env);
            },
            .Try => |try_stmt| {
                try self.resolveBody(try_stmt.try_body, env);
                if (try_stmt.catch_clause) |catch_clause| {
                    var catch_env = try Env.init(self.arena, env);
                    if (catch_clause.error_pattern) |pattern_id| try self.bindPatternIfName(&catch_env, pattern_id);
                    try self.resolveBody(catch_clause.body, &catch_env);
                }
            },
            .Break => |jump| if (jump.value) |expr_id| try self.resolveExpr(expr_id, env),
            .Continue => |jump| if (jump.value) |expr_id| try self.resolveExpr(expr_id, env),
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| try self.resolveExpr(arg, env);
            },
            .Lock => |lock_stmt| try self.resolveExpr(lock_stmt.path, env),
            .Unlock => |unlock_stmt| try self.resolveExpr(unlock_stmt.path, env),
            .Assert => |assert_stmt| try self.resolveExpr(assert_stmt.condition, env),
            .Assume => |assume_stmt| try self.resolveExpr(assume_stmt.condition, env),
            .Havoc => {},
            .Assign => |assign| {
                try self.resolvePattern(assign.target, env);
                try self.resolveExpr(assign.value, env);
            },
            .Expr => |expr_stmt| try self.resolveExpr(expr_stmt.expr, env),
            .Block => |block| try self.resolveBody(block.body, env),
            .LabeledBlock => |block| try self.resolveBody(block.body, env),
            else => {},
        }
    }

    fn resolvePattern(self: *Resolver, pattern_id: ast.PatternId, env: *const Env) anyerror!void {
        switch (self.file.pattern(pattern_id).*) {
            .Field => |field| try self.resolvePattern(field.base, env),
            .Index => |index| {
                try self.resolvePattern(index.base, env);
                try self.resolveExpr(index.index, env);
            },
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| try self.resolvePattern(field.binding, env);
            },
            else => {},
        }
    }

    fn resolveExpr(self: *Resolver, expr_id: ast.ExprId, env: *const Env) anyerror!void {
        switch (self.file.expression(expr_id).*) {
            .Name => |name| {
                self.bindings[expr_id.index()] = env.lookup(name.name);
                if (self.bindings[expr_id.index()] == null and !self.isRecognizedTypeValueName(name.name)) {
                    var buffer: [256]u8 = undefined;
                    const message = try std.fmt.bufPrint(&buffer, "undefined name '{s}'", .{name.name});
                    try self.diagnostics.appendError(message, .{
                        .file_id = self.file_id,
                        .range = name.range,
                    });
                }
            },
            .TypeValue => {},
            .Unary => |unary| try self.resolveExpr(unary.operand, env),
            .Binary => |binary| {
                try self.resolveExpr(binary.lhs, env);
                try self.resolveExpr(binary.rhs, env);
            },
            .Tuple => |tuple| {
                for (tuple.elements) |element| try self.resolveExpr(element, env);
            },
            .ArrayLiteral => |array| {
                for (array.elements) |element| try self.resolveExpr(element, env);
            },
            .StructLiteral => |struct_literal| {
                for (struct_literal.fields) |field| try self.resolveExpr(field.value, env);
            },
            .Switch => |switch_expr| {
                try self.resolveExpr(switch_expr.condition, env);
                for (switch_expr.arms) |arm| {
                    var arm_env = try Env.init(self.arena, env);
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.resolveExpr(pattern_expr, env),
                        .Range => |range_pattern| {
                            try self.resolveExpr(range_pattern.start, env);
                            try self.resolveExpr(range_pattern.end, env);
                        },
                        .Or => |or_pattern| {
                            for (or_pattern.alternatives) |alternative| {
                                try self.resolveSwitchPattern(alternative, env, &arm_env);
                            }
                        },
                        .Ok, .Err => |pattern_id| try self.bindPatternIfName(&arm_env, pattern_id),
                        .NamedError => |named_error| {
                            try self.resolveExpr(named_error.callee, env);
                            for (named_error.bindings) |pattern_id| try self.bindPatternIfName(&arm_env, pattern_id);
                        },
                        .Else => {},
                    }
                    try self.resolveExpr(arm.value, &arm_env);
                }
                if (switch_expr.else_expr) |else_expr| try self.resolveExpr(else_expr, env);
            },
            .Comptime => |comptime_expr| try self.resolveBody(comptime_expr.body, env),
            .ErrorReturn => |error_return| {
                for (error_return.args) |arg| try self.resolveExpr(arg, env);
            },
            .Call => |call| {
                try self.resolveExpr(call.callee, env);
                for (call.args) |arg| try self.resolveExpr(arg, env);
            },
            .Builtin => |builtin| {
                for (builtin.args) |arg| try self.resolveExpr(arg, env);
            },
            .Field => |field| {
                if (self.emitDeprecatedErrorNamespace(expr_id, field)) return;
                try self.resolveExpr(field.base, env);
            },
            .Index => |index| {
                try self.resolveExpr(index.base, env);
                try self.resolveExpr(index.index, env);
            },
            .Group => |group| try self.resolveExpr(group.expr, env),
            .Old => |old| try self.resolveExpr(old.expr, env),
            .Quantified => |quantified| {
                var quant_env = try Env.init(self.arena, env);
                try self.bindPatternIfName(&quant_env, quantified.pattern);
                if (quantified.condition) |condition| try self.resolveExpr(condition, &quant_env);
                try self.resolveExpr(quantified.body, &quant_env);
            },
            else => {},
        }
    }

    fn bindPatternIfName(self: *Resolver, env: *Env, pattern_id: ast.PatternId) !void {
        switch (self.file.pattern(pattern_id).*) {
            .Name => |name| {
                if (std.mem.eql(u8, name.name, "_")) return;
                try env.bindings.put(name.name, .{ .pattern = pattern_id });
            },
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| try self.bindPatternIfName(env, field.binding);
            },
            else => {},
        }
    }

    fn firstImplSelfPattern(self: *Resolver, impl_item: ast.ImplItem) ?ast.PatternId {
        for (impl_item.methods) |method_id| {
            const item = self.file.item(method_id).*;
            if (item != .Function) continue;
            for (item.Function.parameters) |parameter| {
                if (parameter.is_comptime) continue;
                const pattern = self.file.pattern(parameter.pattern).*;
                if (pattern != .Name) break;
                if (std.mem.eql(u8, pattern.Name.name, "self")) return parameter.pattern;
                break;
            }
        }
        return null;
    }

    fn lookupTraitGhostItem(self: *Resolver, trait_name: []const u8, env: *const Env) ?ast.ItemId {
        const binding = env.lookup(trait_name) orelse return null;
        const item_id = switch (binding) {
            .item => |item_id| item_id,
            .pattern => return null,
        };
        const item = self.file.item(item_id).*;
        if (item != .Trait) return null;
        return item.Trait.ghost_block;
    }

    fn emitDeprecatedErrorNamespace(self: *Resolver, expr_id: ast.ExprId, field: ast.FieldExpr) bool {
        _ = expr_id;
        const base_name = switch (self.file.expression(field.base).*) {
            .Name => |name| name.name,
            .Group => |group| switch (self.file.expression(group.expr).*) {
                .Name => |name| name.name,
                else => return false,
            },
            else => return false,
        };
        if (!std.mem.eql(u8, base_name, "error")) return false;
        var buffer: [256]u8 = undefined;
        const message = std.fmt.bufPrint(&buffer, "use '{s}' or '{s}(...)' instead of 'error.{s}'", .{
            field.name,
            field.name,
            field.name,
        }) catch "error values use 'Name' or 'Name(...)', not 'error.Name'";
        self.diagnostics.appendError(message, .{
            .file_id = self.file_id,
            .range = field.range,
        }) catch {};
        return true;
    }

    fn isRecognizedTypeValueName(self: *const Resolver, name: []const u8) bool {
        const trimmed = std.mem.trim(u8, name, " \t\n\r");
        if (std.mem.eql(u8, trimmed, "void") or
            std.mem.eql(u8, trimmed, "bool") or
            std.mem.eql(u8, trimmed, "string") or
            std.mem.eql(u8, trimmed, "address") or
            std.mem.eql(u8, trimmed, "bytes") or
            std.mem.eql(u8, trimmed, "std") or
            std.mem.eql(u8, trimmed, "Ok") or
            std.mem.eql(u8, trimmed, "Err"))
        {
            return true;
        }

        if (trimmed.len >= 2 and (trimmed[0] == 'u' or trimmed[0] == 'i')) {
            const bits = trimmed[1..];
            if (std.mem.eql(u8, bits, "8") or
                std.mem.eql(u8, bits, "16") or
                std.mem.eql(u8, bits, "32") or
                std.mem.eql(u8, bits, "64") or
                std.mem.eql(u8, bits, "128") or
                std.mem.eql(u8, bits, "256"))
            {
                return true;
            }
        }

        if (self.lookupTypeValueItem(trimmed)) |_| return true;
        return false;
    }

    fn lookupTypeValueItem(self: *const Resolver, name: []const u8) ?ast.ItemId {
        const item_id = self.item_index.lookup(name) orelse return null;
        return switch (self.file.item(item_id).*) {
            .Contract, .Struct, .Bitfield, .Enum, .TypeAlias => item_id,
            else => null,
        };
    }
};
