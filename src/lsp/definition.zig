const std = @import("std");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");

const Allocator = std.mem.Allocator;

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

const file_origin: frontend.Range = .{
    .start = .{ .line = 0, .character = 0 },
    .end = .{ .line = 0, .character = 0 },
};

pub const Analysis = struct {
    sources: compiler.source.SourceStore,
    file_id: compiler.FileId,
    module_id: compiler.ModuleId,
    parse_result: compiler.syntax.ParseResult,
    lower_result: compiler.ast.LowerResult,
    item_index: compiler.sema.ItemIndexResult,
    resolution: compiler.sema.NameResolutionResult,
    source_text: []const u8,

    pub fn init(allocator: Allocator, source_text: []const u8) !?Analysis {
        var sources = compiler.source.SourceStore.init(allocator);
        errdefer sources.deinit();
        const file_id = try sources.addFile("<lsp>", source_text);
        const package_id = try sources.addPackage("main");
        const module_id = try sources.addModule(package_id, file_id, "lsp");

        var parse_result = try compiler.syntax.parse(allocator, file_id, source_text);
        errdefer parse_result.deinit();

        var lower_result = try compiler.ast.lower(allocator, &parse_result.tree);
        errdefer lower_result.deinit();

        var item_index = try compiler.sema.buildItemIndex(allocator, &lower_result.file);
        errdefer item_index.deinit();
        var resolution = try compiler.sema.resolveNames(allocator, file_id, &lower_result.file, &item_index);
        errdefer resolution.deinit();

        return .{
            .sources = sources,
            .file_id = file_id,
            .module_id = module_id,
            .parse_result = parse_result,
            .lower_result = lower_result,
            .item_index = item_index,
            .resolution = resolution,
            .source_text = source_text,
        };
    }

    pub fn deinit(self: *Analysis) void {
        self.resolution.deinit();
        self.item_index.deinit();
        self.lower_result.deinit();
        self.parse_result.deinit();
        self.sources.deinit();
    }

    fn textRangeToRange(self: *const Analysis, range: compiler.TextRange) frontend.Range {
        const start = self.sources.lineColumn(.{
            .file_id = self.file_id,
            .range = .{ .start = range.start, .end = range.start },
        });
        const end = self.sources.lineColumn(.{
            .file_id = self.file_id,
            .range = .{ .start = range.end, .end = range.end },
        });
        return .{
            .start = .{
                .line = if (start.line > 0) start.line - 1 else 0,
                .character = if (start.column > 0) start.column - 1 else 0,
            },
            .end = .{
                .line = if (end.line > 0) end.line - 1 else 0,
                .character = if (end.column > 0) end.column - 1 else 0,
            },
        };
    }

    fn file(self: *const Analysis) *const compiler.ast.AstFile {
        return &self.lower_result.file;
    }

    fn selectionRange(self: *const Analysis, range: compiler.TextRange, name: []const u8) frontend.Range {
        var name_start = range.start;
        const start: usize = @intCast(@min(range.start, self.source_text.len));
        const end: usize = @intCast(@min(range.end, self.source_text.len));
        if (start <= end and end <= self.source_text.len) {
            if (std.mem.indexOf(u8, self.source_text[start..end], name)) |relative| {
                const relative_u32 = std.math.cast(u32, relative) orelse 0;
                name_start = std.math.add(u32, range.start, relative_u32) catch range.start;
            }
        }
        return self.textRangeToRange(.{
            .start = name_start,
            .end = name_start + @as(u32, @intCast(name.len)),
        });
    }
};

const Resolver = struct {
    analysis: *const Analysis,
    query_position: frontend.Position,
    cross_file: ?CrossFileContext,
    found: ?Definition = null,

    fn run(self: *Resolver) anyerror!?Definition {
        for (self.analysis.file().root_items) |item_id| {
            if (try self.visitItem(item_id)) break;
        }
        return self.found;
    }

    fn visitItem(self: *Resolver, item_id: compiler.ast.ItemId) anyerror!bool {
        if (self.found != null) return true;
        const item = self.analysis.file().item(item_id).*;

        if (self.matchItemDeclaration(item_id)) return true;

        switch (item) {
            .Contract => |contract_decl| {
                for (contract_decl.invariants) |expr_id| {
                    if (try self.visitExpr(expr_id)) return true;
                }
                for (contract_decl.members) |member_id| {
                    if (try self.visitItem(member_id)) return true;
                }
            },
            .Function => |function_decl| {
                for (function_decl.parameters) |parameter| {
                    if (self.matchPatternDeclaration(parameter.pattern)) return true;
                }
                for (function_decl.clauses) |clause| {
                    if (try self.visitExpr(clause.expr)) return true;
                }
                if (try self.visitBody(function_decl.body)) return true;
            },
            .Field => |field_decl| {
                if (field_decl.value) |value| {
                    if (try self.visitExpr(value)) return true;
                }
            },
            .Constant => |constant_decl| {
                if (try self.visitExpr(constant_decl.value)) return true;
            },
            .Trait => |trait_decl| {
                for (trait_decl.methods) |method| {
                    if (self.matchNameRange(method.range, method.name)) return true;
                    for (method.parameters) |parameter| {
                        if (self.matchPatternDeclaration(parameter.pattern)) return true;
                    }
                    for (method.clauses) |clause| {
                        if (try self.visitExpr(clause.expr)) return true;
                    }
                }
                if (trait_decl.ghost_block) |ghost_id| {
                    if (try self.visitItem(ghost_id)) return true;
                }
            },
            .GhostBlock => |ghost_block| {
                if (try self.visitBody(ghost_block.body)) return true;
            },
            .Struct => |struct_decl| {
                for (struct_decl.fields) |field| {
                    if (self.matchNameRange(field.range, field.name)) return true;
                }
            },
            .Bitfield => |bitfield_decl| {
                for (bitfield_decl.fields) |field| {
                    if (self.matchNameRange(field.range, field.name)) return true;
                }
            },
            .Enum => |enum_decl| {
                for (enum_decl.variants) |variant| {
                    if (self.matchNameRange(variant.range, variant.name)) return true;
                }
            },
            .LogDecl => |log_decl| {
                for (log_decl.fields) |field| {
                    if (self.matchNameRange(field.range, field.name)) return true;
                }
            },
            .ErrorDecl => |error_decl| {
                for (error_decl.parameters) |parameter| {
                    if (self.matchPatternDeclaration(parameter.pattern)) return true;
                }
            },
            .Impl => |impl_decl| {
                for (impl_decl.methods) |method_id| {
                    if (try self.visitItem(method_id)) return true;
                }
            },
            else => {},
        }

        return false;
    }

    fn visitBody(self: *Resolver, body_id: compiler.ast.BodyId) anyerror!bool {
        const body = self.analysis.file().body(body_id).*;
        for (body.statements) |stmt_id| {
            if (try self.visitStmt(stmt_id)) return true;
        }
        return false;
    }

    fn visitStmt(self: *Resolver, stmt_id: compiler.ast.StmtId) anyerror!bool {
        if (self.found != null) return true;
        switch (self.analysis.file().statement(stmt_id).*) {
            .VariableDecl => |decl| {
                if (decl.value) |expr_id| {
                    if (try self.visitExpr(expr_id)) return true;
                }
                if (self.matchPatternDeclaration(decl.pattern)) return true;
            },
            .Return => |ret| if (ret.value) |expr_id| {
                if (try self.visitExpr(expr_id)) return true;
            },
            .If => |if_stmt| {
                if (try self.visitExpr(if_stmt.condition)) return true;
                if (try self.visitBody(if_stmt.then_body)) return true;
                if (if_stmt.else_body) |else_body| {
                    if (try self.visitBody(else_body)) return true;
                }
            },
            .While => |while_stmt| {
                if (try self.visitExpr(while_stmt.condition)) return true;
                for (while_stmt.invariants) |expr_id| {
                    if (try self.visitExpr(expr_id)) return true;
                }
                if (try self.visitBody(while_stmt.body)) return true;
            },
            .For => |for_stmt| {
                if (try self.visitExpr(for_stmt.iterable)) return true;
                if (self.matchPatternDeclaration(for_stmt.item_pattern)) return true;
                if (for_stmt.index_pattern) |index_pattern| {
                    if (self.matchPatternDeclaration(index_pattern)) return true;
                }
                for (for_stmt.invariants) |expr_id| {
                    if (try self.visitExpr(expr_id)) return true;
                }
                if (try self.visitBody(for_stmt.body)) return true;
            },
            .Switch => |switch_stmt| {
                if (try self.visitExpr(switch_stmt.condition)) return true;
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| if (try self.visitExpr(expr_id)) return true,
                        .Range => |range_pattern| {
                            if (try self.visitExpr(range_pattern.start)) return true;
                            if (try self.visitExpr(range_pattern.end)) return true;
                        },
                        .Ok, .Err => |pattern_id| if (self.matchPatternDeclaration(pattern_id)) return true,
                        .Else => {},
                    }
                    if (try self.visitBody(arm.body)) return true;
                }
                if (switch_stmt.else_body) |else_body| {
                    if (try self.visitBody(else_body)) return true;
                }
            },
            .Try => |try_stmt| {
                if (try self.visitBody(try_stmt.try_body)) return true;
                if (try_stmt.catch_clause) |catch_clause| {
                    if (catch_clause.error_pattern) |pattern_id| {
                        if (self.matchPatternDeclaration(pattern_id)) return true;
                    }
                    if (try self.visitBody(catch_clause.body)) return true;
                }
            },
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| {
                    if (try self.visitExpr(arg)) return true;
                }
            },
            .Lock => |lock_stmt| if (try self.visitExpr(lock_stmt.path)) return true,
            .Unlock => |unlock_stmt| if (try self.visitExpr(unlock_stmt.path)) return true,
            .Assert => |assert_stmt| if (try self.visitExpr(assert_stmt.condition)) return true,
            .Assume => |assume_stmt| if (try self.visitExpr(assume_stmt.condition)) return true,
            .Break => |jump| if (jump.value) |expr_id| {
                if (try self.visitExpr(expr_id)) return true;
            },
            .Continue => |jump| if (jump.value) |expr_id| {
                if (try self.visitExpr(expr_id)) return true;
            },
            .Assign => |assign| {
                if (self.matchPatternDeclaration(assign.target)) return true;
                if (try self.visitPatternUses(assign.target)) return true;
                if (try self.visitExpr(assign.value)) return true;
            },
            .Expr => |expr_stmt| if (try self.visitExpr(expr_stmt.expr)) return true,
            .Block => |block| if (try self.visitBody(block.body)) return true,
            .LabeledBlock => |block| if (try self.visitBody(block.body)) return true,
            else => {},
        }
        return false;
    }

    fn visitPatternUses(self: *Resolver, pattern_id: compiler.ast.PatternId) anyerror!bool {
        switch (self.analysis.file().pattern(pattern_id).*) {
            .Field => |field| return self.visitPatternUses(field.base),
            .Index => |index| {
                if (try self.visitPatternUses(index.base)) return true;
                if (try self.visitExpr(index.index)) return true;
            },
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| {
                    if (self.matchNameRange(field.range, field.name)) return true;
                    if (try self.visitPatternUses(field.binding)) return true;
                }
            },
            else => {},
        }
        return false;
    }

    fn visitExpr(self: *Resolver, expr_id: compiler.ast.ExprId) anyerror!bool {
        if (self.found != null) return true;
        const expr = self.analysis.file().expression(expr_id).*;
        switch (expr) {
            .Name => |name| {
                if (self.touchesNameRange(name.range, name.name)) {
                    if (self.resolveImportAliasName(name.name)) return true;
                    return try self.resolveExprBinding(expr_id);
                }
            },
            .Unary => |unary| if (try self.visitExpr(unary.operand)) return true,
            .Binary => |binary| {
                if (try self.visitExpr(binary.lhs)) return true;
                if (try self.visitExpr(binary.rhs)) return true;
            },
            .Tuple => |tuple| {
                for (tuple.elements) |element| {
                    if (try self.visitExpr(element)) return true;
                }
            },
            .ArrayLiteral => |array| {
                for (array.elements) |element| {
                    if (try self.visitExpr(element)) return true;
                }
            },
            .StructLiteral => |struct_literal| {
                for (struct_literal.fields) |field| {
                    if (self.matchNameRange(field.range, field.name)) return true;
                    if (try self.visitExpr(field.value)) return true;
                }
            },
            .Switch => |switch_expr| {
                if (try self.visitExpr(switch_expr.condition)) return true;
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| if (try self.visitExpr(pattern_expr)) return true,
                        .Range => |range_pattern| {
                            if (try self.visitExpr(range_pattern.start)) return true;
                            if (try self.visitExpr(range_pattern.end)) return true;
                        },
                        .Ok, .Err => |pattern_id| if (self.matchPatternDeclaration(pattern_id)) return true,
                        .Else => {},
                    }
                    if (try self.visitExpr(arm.value)) return true;
                }
                if (switch_expr.else_expr) |else_expr| {
                    if (try self.visitExpr(else_expr)) return true;
                }
            },
            .Comptime => |comptime_expr| if (try self.visitBody(comptime_expr.body)) return true,
            .ErrorReturn => |error_return| {
                if (self.touchesNameRange(error_return.range, error_return.name)) return false;
                for (error_return.args) |arg| {
                    if (try self.visitExpr(arg)) return true;
                }
            },
            .Call => |call| {
                if (try self.visitExpr(call.callee)) return true;
                for (call.args) |arg| {
                    if (try self.visitExpr(arg)) return true;
                }
            },
            .Builtin => |builtin| {
                if (self.touchesNameRange(builtin.range, builtin.name)) return false;
                for (builtin.args) |arg| {
                    if (try self.visitExpr(arg)) return true;
                }
            },
            .Field => |field| {
                if (self.touchesNameRange(field.range, field.name)) {
                    if (try self.resolveCrossFileField(expr_id, field)) return true;
                }
                if (try self.visitExpr(field.base)) return true;
            },
            .Index => |index| {
                if (try self.visitExpr(index.base)) return true;
                if (try self.visitExpr(index.index)) return true;
            },
            .Group => |group| if (try self.visitExpr(group.expr)) return true,
            .Old => |old| if (try self.visitExpr(old.expr)) return true,
            .Quantified => |quantified| {
                if (self.matchPatternDeclaration(quantified.pattern)) return true;
                if (quantified.condition) |condition| {
                    if (try self.visitExpr(condition)) return true;
                }
                if (try self.visitExpr(quantified.body)) return true;
            },
            else => {},
        }
        return false;
    }

    fn resolveExprBinding(self: *Resolver, expr_id: compiler.ast.ExprId) anyerror!bool {
        const binding = self.analysis.resolution.expr_bindings[expr_id.index()] orelse return false;
        self.found = try self.definitionForBinding(binding);
        return self.found != null;
    }

    fn resolveImportAliasName(self: *Resolver, name: []const u8) bool {
        const item_id = self.importAliasItemByName(name) orelse return false;
        self.found = self.definitionForItem(item_id) catch null;
        return self.found != null;
    }

    fn definitionForBinding(self: *Resolver, binding: compiler.sema.ResolvedBinding) anyerror!?Definition {
        return switch (binding) {
            .item => |item_id| try self.definitionForItem(item_id),
            .pattern => |pattern_id| self.definitionForPattern(pattern_id),
        };
    }

    fn definitionForItem(self: *Resolver, item_id: compiler.ast.ItemId) anyerror!?Definition {
        if (self.itemImportAliasName(item_id)) |alias| {
            if (findImportBinding(self.cross_file, alias)) |binding| {
                return .{ .uri = binding.target_uri, .range = file_origin };
            }
        }

        const range = self.itemSelectionRange(item_id) orelse return null;
        return .{ .range = range };
    }

    fn importAliasItemByName(self: *Resolver, alias: []const u8) ?compiler.ast.ItemId {
        for (self.analysis.file().root_items) |item_id| {
            const item_alias = self.itemImportAliasName(item_id) orelse continue;
            if (std.mem.eql(u8, item_alias, alias)) return item_id;
        }
        return null;
    }

    fn itemImportAliasName(self: *Resolver, item_id: compiler.ast.ItemId) ?[]const u8 {
        const item = self.analysis.file().item(item_id).*;
        return switch (item) {
            .Import => |import_item| import_item.alias,
            .Constant => |constant_item| blk: {
                switch (self.analysis.file().expression(constant_item.value).*) {
                    .Builtin => |builtin| if (std.mem.eql(u8, builtin.name, "import")) break :blk constant_item.name,
                    else => {},
                }
                const start: usize = @intCast(@min(constant_item.range.start, self.analysis.source_text.len));
                const end: usize = @intCast(@min(constant_item.range.end, self.analysis.source_text.len));
                if (start <= end and end <= self.analysis.source_text.len and std.mem.indexOf(u8, self.analysis.source_text[start..end], "@import(") != null) {
                    break :blk constant_item.name;
                }
                break :blk null;
            },
            else => null,
        };
    }

    fn definitionForPattern(self: *Resolver, pattern_id: compiler.ast.PatternId) ?Definition {
        const range = self.patternSelectionRange(pattern_id) orelse return null;
        return .{ .range = range };
    }

    fn resolveCrossFileField(self: *Resolver, expr_id: compiler.ast.ExprId, field: compiler.ast.FieldExpr) anyerror!bool {
        const binding = switch (self.analysis.file().expression(field.base).*) {
            .Name => self.analysis.resolution.expr_bindings[field.base.index()],
            .Group => |group| switch (self.analysis.file().expression(group.expr).*) {
                .Name => self.analysis.resolution.expr_bindings[group.expr.index()],
                else => null,
            },
            else => null,
        } orelse return false;

        const item_id = switch (binding) {
            .item => |item_id| item_id,
            .pattern => return false,
        };
        const item = self.analysis.file().item(item_id).*;
        if (item != .Import) return false;

        const alias = item.Import.alias orelse return false;
        const import_binding = findImportBinding(self.cross_file, alias) orelse return false;

        if (import_binding.target_source) |target_source| {
            if (try findTopLevelDeclarationInSource(self.analysis.sources.allocator, target_source, field.name)) |range| {
                self.found = .{
                    .uri = import_binding.target_uri,
                    .range = range,
                };
                return true;
            }
        }

        self.found = .{
            .uri = import_binding.target_uri,
            .range = file_origin,
        };
        _ = expr_id;
        return true;
    }

    fn matchItemDeclaration(self: *Resolver, item_id: compiler.ast.ItemId) bool {
        const range = self.itemSelectionRange(item_id) orelse return false;
        if (!rangeContainsPosition(range, self.query_position)) return false;
        self.found = .{ .range = range };
        return true;
    }

    fn matchPatternDeclaration(self: *Resolver, pattern_id: compiler.ast.PatternId) bool {
        const range = self.patternSelectionRange(pattern_id) orelse return false;
        if (!rangeContainsPosition(range, self.query_position)) return false;
        self.found = .{ .range = range };
        return true;
    }

    fn matchNameRange(self: *Resolver, range: compiler.TextRange, name: []const u8) bool {
        const selection = self.analysis.selectionRange(range, name);
        if (!rangeContainsPosition(selection, self.query_position)) return false;
        self.found = .{ .range = selection };
        return true;
    }

    fn touchesNameRange(self: *Resolver, range: compiler.TextRange, name: []const u8) bool {
        return rangeContainsPosition(self.analysis.selectionRange(range, name), self.query_position);
    }

    fn itemSelectionRange(self: *Resolver, item_id: compiler.ast.ItemId) ?frontend.Range {
        const item = self.analysis.file().item(item_id).*;
        const pair: ?struct { range: compiler.TextRange, name: []const u8 } = switch (item) {
            .Import => |import_item| if (import_item.alias) |alias|
                .{ .range = import_item.range, .name = alias }
            else
                null,
            .Contract => |contract_decl| .{ .range = contract_decl.range, .name = contract_decl.name },
            .Function => |function_decl| .{ .range = function_decl.range, .name = function_decl.name },
            .Struct => |struct_decl| .{ .range = struct_decl.range, .name = struct_decl.name },
            .Bitfield => |bitfield_decl| .{ .range = bitfield_decl.range, .name = bitfield_decl.name },
            .Enum => |enum_decl| .{ .range = enum_decl.range, .name = enum_decl.name },
            .Trait => |trait_decl| .{ .range = trait_decl.range, .name = trait_decl.name },
            .TypeAlias => |alias_decl| .{ .range = alias_decl.range, .name = alias_decl.name },
            .LogDecl => |log_decl| .{ .range = log_decl.range, .name = log_decl.name },
            .ErrorDecl => |error_decl| .{ .range = error_decl.range, .name = error_decl.name },
            .Field => |field_decl| .{ .range = field_decl.range, .name = field_decl.name },
            .Constant => |constant_decl| .{ .range = constant_decl.range, .name = constant_decl.name },
            else => null,
        };
        if (pair) |p| return self.analysis.selectionRange(p.range, p.name);
        return null;
    }

    fn patternSelectionRange(self: *Resolver, pattern_id: compiler.ast.PatternId) ?frontend.Range {
        const pattern = self.analysis.file().pattern(pattern_id).*;
        return switch (pattern) {
            .Name => |name| self.analysis.selectionRange(name.range, name.name),
            .StructDestructure => |destructure| self.analysis.textRangeToRange(destructure.range),
            else => null,
        };
    }
};

pub fn definitionAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Definition {
    return definitionAtImpl(allocator, source, position, null);
}

pub fn definitionAtCrossFile(allocator: Allocator, source: []const u8, position: frontend.Position, cross_file: CrossFileContext) !?Definition {
    return definitionAtImpl(allocator, source, position, cross_file);
}

/// Resolve definition using a pre-built analysis (avoids re-parsing).
pub fn definitionAtCached(analysis: *Analysis, source: []const u8, position: frontend.Position) ?Definition {
    var resolver = Resolver{
        .analysis = analysis,
        .query_position = position,
        .cross_file = null,
    };
    _ = source;
    return resolver.run() catch null;
}

fn definitionAtImpl(allocator: Allocator, source: []const u8, position: frontend.Position, cross_file: ?CrossFileContext) !?Definition {
    var analysis = (try Analysis.init(allocator, source)) orelse return null;
    defer analysis.deinit();

    if (identifierAtPosition(source, position)) |name| {
        if (findImportBinding(cross_file, name)) |binding| {
            if (hasTopLevelDeclarationNamed(&analysis, name)) {
                if (binding.target_source) |target_source| {
                    if (nextFieldIdentifier(source, position)) |member_name| {
                        if (try findTopLevelDeclarationInSource(allocator, target_source, member_name)) |range| {
                            return .{
                                .uri = binding.target_uri,
                                .range = range,
                            };
                        }
                    }
                }
                return .{
                    .uri = binding.target_uri,
                    .range = file_origin,
                };
            }
        }

        for (analysis.file().root_items) |item_id| {
            var alias_resolver = Resolver{
                .analysis = &analysis,
                .query_position = position,
                .cross_file = cross_file,
            };
            const alias = alias_resolver.itemImportAliasName(item_id) orelse continue;
            if (!std.mem.eql(u8, alias, name)) continue;
            return try alias_resolver.definitionForItem(item_id);
        }
    }

    var resolver = Resolver{
        .analysis = &analysis,
        .query_position = position,
        .cross_file = cross_file,
    };
    return resolver.run();
}

fn findImportBinding(cross_file: ?CrossFileContext, name: []const u8) ?ImportBinding {
    const cf = cross_file orelse return null;
    for (cf.bindings) |binding| {
        if (std.mem.eql(u8, binding.alias, name)) return binding;
    }
    return null;
}

fn findTopLevelDeclarationInSource(allocator: Allocator, source_text: []const u8, name: []const u8) !?frontend.Range {
    var analysis = (try Analysis.init(allocator, source_text)) orelse return null;
    defer analysis.deinit();

    for (analysis.file().root_items) |item_id| {
        const item = analysis.file().item(item_id).*;
        const pair: ?struct { range: compiler.TextRange, symbol: []const u8 } = switch (item) {
            .Contract => |contract_decl| .{ .range = contract_decl.range, .symbol = contract_decl.name },
            .Function => |function_decl| .{ .range = function_decl.range, .symbol = function_decl.name },
            .Field => |field_decl| .{ .range = field_decl.range, .symbol = field_decl.name },
            .Constant => |constant_decl| .{ .range = constant_decl.range, .symbol = constant_decl.name },
            .Struct => |struct_decl| .{ .range = struct_decl.range, .symbol = struct_decl.name },
            .Bitfield => |bitfield_decl| .{ .range = bitfield_decl.range, .symbol = bitfield_decl.name },
            .Enum => |enum_decl| .{ .range = enum_decl.range, .symbol = enum_decl.name },
            .Trait => |trait_decl| .{ .range = trait_decl.range, .symbol = trait_decl.name },
            .TypeAlias => |alias_decl| .{ .range = alias_decl.range, .symbol = alias_decl.name },
            .LogDecl => |log_decl| .{ .range = log_decl.range, .symbol = log_decl.name },
            .ErrorDecl => |error_decl| .{ .range = error_decl.range, .symbol = error_decl.name },
            else => null,
        };
        if (pair) |p| {
            if (std.mem.eql(u8, p.symbol, name)) {
                return analysis.selectionRange(p.range, p.symbol);
            }
        }
    }

    return null;
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

fn identifierAtPosition(source_text: []const u8, position: frontend.Position) ?[]const u8 {
    const byte_index = byteIndexFromPosition(source_text, position) orelse return null;
    if (byte_index >= source_text.len) return null;
    if (!isIdentifierChar(source_text[byte_index])) return null;

    var start = byte_index;
    while (start > 0 and isIdentifierChar(source_text[start - 1])) start -= 1;

    var end = byte_index;
    while (end < source_text.len and isIdentifierChar(source_text[end])) end += 1;

    return source_text[start..end];
}

fn nextFieldIdentifier(source_text: []const u8, position: frontend.Position) ?[]const u8 {
    const byte_index = byteIndexFromPosition(source_text, position) orelse return null;
    if (byte_index >= source_text.len or !isIdentifierChar(source_text[byte_index])) return null;

    var end = byte_index;
    while (end < source_text.len and isIdentifierChar(source_text[end])) end += 1;
    if (end >= source_text.len or source_text[end] != '.') return null;
    const member_start = end + 1;
    if (member_start >= source_text.len or !isIdentifierChar(source_text[member_start])) return null;
    var member_end = member_start;
    while (member_end < source_text.len and isIdentifierChar(source_text[member_end])) member_end += 1;
    return source_text[member_start..member_end];
}

fn byteIndexFromPosition(source_text: []const u8, position: frontend.Position) ?usize {
    var line: u32 = 0;
    var character: u32 = 0;
    for (source_text, 0..) |byte, index| {
        if (line == position.line and character == position.character) return index;
        if (byte == '\n') {
            line += 1;
            character = 0;
        } else {
            character += 1;
        }
    }
    if (line == position.line and character == position.character) return source_text.len;
    return null;
}

fn isIdentifierChar(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

fn hasTopLevelDeclarationNamed(analysis: *const Analysis, name: []const u8) bool {
    for (analysis.file().root_items) |item_id| {
        const item = analysis.file().item(item_id).*;
        const item_name: ?[]const u8 = switch (item) {
            .Import => |import_item| import_item.alias,
            .Contract => |contract_decl| contract_decl.name,
            .Function => |function_decl| function_decl.name,
            .Field => |field_decl| field_decl.name,
            .Constant => |constant_decl| constant_decl.name,
            .Struct => |struct_decl| struct_decl.name,
            .Bitfield => |bitfield_decl| bitfield_decl.name,
            .Enum => |enum_decl| enum_decl.name,
            .Trait => |trait_decl| trait_decl.name,
            .TypeAlias => |alias_decl| alias_decl.name,
            .LogDecl => |log_decl| log_decl.name,
            .ErrorDecl => |error_decl| error_decl.name,
            else => null,
        };
        if (item_name) |item_name_slice| {
            if (std.mem.eql(u8, item_name_slice, name)) return true;
        }
    }
    return false;
}
