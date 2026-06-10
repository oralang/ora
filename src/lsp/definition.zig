const std = @import("std");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");

const Allocator = std.mem.Allocator;

pub const Definition = struct {
    uri: ?[]const u8 = null,
    range: frontend.Range,
};

pub const ResolvedName = struct {
    name: []const u8,
    range: frontend.Range,
    definition_range: frontend.Range,
};

pub const ImportBinding = struct {
    alias: []const u8,
    target_uri: []const u8,
};

pub const CrossFileContext = struct {
    bindings: []const ImportBinding,
};

const file_origin: frontend.Range = .{
    .start = .{ .line = 0, .character = 0 },
    .end = .{ .line = 0, .character = 0 },
};

pub const Analysis = struct {
    allocator: Allocator,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    module_id: compiler.ModuleId,
    ast_file: *const compiler.ast.AstFile,
    resolution: *const compiler.sema.NameResolutionResult,
    source_text: []const u8,

    pub fn initBorrowed(
        allocator: Allocator,
        sources: *const compiler.source.SourceStore,
        file_id: compiler.FileId,
        module_id: compiler.ModuleId,
        source_text: []const u8,
        ast_file: *const compiler.ast.AstFile,
        item_index: *const compiler.sema.ItemIndexResult,
        resolution: *const compiler.sema.NameResolutionResult,
    ) Analysis {
        _ = item_index;
        return .{
            .allocator = allocator,
            .sources = sources,
            .file_id = file_id,
            .module_id = module_id,
            .ast_file = ast_file,
            .resolution = resolution,
            .source_text = source_text,
        };
    }

    pub fn deinit(self: *Analysis) void {
        self.* = undefined;
    }

    fn textRangeToRange(self: *const Analysis, range: compiler.TextRange) frontend.Range {
        const sources = self.sourceStore();
        const start = sources.lineColumn(.{
            .file_id = self.file_id,
            .range = .{ .start = range.start, .end = range.start },
        });
        const end = sources.lineColumn(.{
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

    fn astFile(self: *const Analysis) *const compiler.ast.AstFile {
        return self.ast_file;
    }

    fn resolutionView(self: *const Analysis) *const compiler.sema.NameResolutionResult {
        return self.resolution;
    }

    fn sourceStore(self: *const Analysis) *const compiler.source.SourceStore {
        return self.sources;
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
        for (self.analysis.astFile().root_items) |item_id| {
            if (try self.visitItem(item_id)) break;
        }
        return self.found;
    }

    fn visitItem(self: *Resolver, item_id: compiler.ast.ItemId) anyerror!bool {
        if (self.found != null) return true;
        const item = self.analysis.astFile().item(item_id).*;

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
        const body = self.analysis.astFile().body(body_id).*;
        for (body.statements) |stmt_id| {
            if (try self.visitStmt(stmt_id)) return true;
        }
        return false;
    }

    fn visitStmt(self: *Resolver, stmt_id: compiler.ast.StmtId) anyerror!bool {
        if (self.found != null) return true;
        switch (self.analysis.astFile().statement(stmt_id).*) {
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
                    if (try self.visitSwitchPattern(arm.pattern)) return true;
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
        switch (self.analysis.astFile().pattern(pattern_id).*) {
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
        const expr = self.analysis.astFile().expression(expr_id).*;
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
                    if (try self.visitSwitchPattern(arm.pattern)) return true;
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

    fn visitSwitchPattern(self: *Resolver, pattern: compiler.ast.SwitchPattern) anyerror!bool {
        switch (pattern) {
            .Expr => |expr_id| if (try self.visitExpr(expr_id)) return true,
            .Range => |range_pattern| {
                if (try self.visitExpr(range_pattern.start)) return true;
                if (try self.visitExpr(range_pattern.end)) return true;
            },
            .NamedError => |named_error| {
                if (try self.visitExpr(named_error.callee)) return true;
                for (named_error.bindings) |pattern_id| {
                    if (self.matchPatternDeclaration(pattern_id)) return true;
                }
            },
            .Or => |or_pattern| {
                for (or_pattern.alternatives) |alternative| {
                    if (try self.visitSwitchPattern(alternative)) return true;
                }
            },
            .Ok, .Err => |pattern_id| if (self.matchPatternDeclaration(pattern_id)) return true,
            .Else => {},
        }
        return false;
    }

    fn resolveExprBinding(self: *Resolver, expr_id: compiler.ast.ExprId) anyerror!bool {
        const binding = self.analysis.resolutionView().expr_bindings[expr_id.index()] orelse return false;
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
        for (self.analysis.astFile().root_items) |item_id| {
            const item_alias = self.itemImportAliasName(item_id) orelse continue;
            if (std.mem.eql(u8, item_alias, alias)) return item_id;
        }
        return null;
    }

    fn itemImportAliasName(self: *Resolver, item_id: compiler.ast.ItemId) ?[]const u8 {
        const item = self.analysis.astFile().item(item_id).*;
        return switch (item) {
            .Import => |import_item| import_item.alias,
            .Constant => |constant_item| blk: {
                switch (self.analysis.astFile().expression(constant_item.value).*) {
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
        const pair = self.itemSelectionPair(item_id) orelse return null;
        return pair.range;
    }

    fn itemSelectionPair(self: *Resolver, item_id: compiler.ast.ItemId) ?struct { range: frontend.Range, name: []const u8 } {
        const item = self.analysis.astFile().item(item_id).*;
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
        if (pair) |p| return .{ .range = self.analysis.selectionRange(p.range, p.name), .name = p.name };
        return null;
    }

    fn patternSelectionRange(self: *Resolver, pattern_id: compiler.ast.PatternId) ?frontend.Range {
        const pair = self.patternSelectionPair(pattern_id) orelse return null;
        return pair.range;
    }

    fn patternSelectionPair(self: *Resolver, pattern_id: compiler.ast.PatternId) ?struct { range: frontend.Range, name: ?[]const u8 } {
        const pattern = self.analysis.astFile().pattern(pattern_id).*;
        return switch (pattern) {
            .Name => |name| .{ .range = self.analysis.selectionRange(name.range, name.name), .name = name.name },
            .StructDestructure => |destructure| .{ .range = self.analysis.textRangeToRange(destructure.range), .name = null },
            else => null,
        };
    }
};

const DefinitionCollector = struct {
    allocator: Allocator,
    resolver: Resolver,
    names: std.ArrayList(ResolvedName) = .{},

    fn init(allocator: Allocator, analysis: *const Analysis) DefinitionCollector {
        return .{
            .allocator = allocator,
            .resolver = .{
                .analysis = analysis,
                .query_position = .{ .line = 0, .character = 0 },
                .cross_file = null,
            },
        };
    }

    fn run(self: *DefinitionCollector) ![]ResolvedName {
        errdefer self.names.deinit(self.allocator);
        for (self.resolver.analysis.astFile().root_items) |item_id| {
            try self.visitItem(item_id);
        }
        return self.names.toOwnedSlice(self.allocator);
    }

    fn visitItem(self: *DefinitionCollector, item_id: compiler.ast.ItemId) anyerror!void {
        try self.appendItemDeclaration(item_id);
        const item = self.resolver.analysis.astFile().item(item_id).*;

        switch (item) {
            .Contract => |contract_decl| {
                for (contract_decl.invariants) |expr_id| try self.visitExpr(expr_id);
                for (contract_decl.members) |member_id| try self.visitItem(member_id);
            },
            .Function => |function_decl| {
                for (function_decl.parameters) |parameter| try self.appendPatternDeclaration(parameter.pattern);
                for (function_decl.clauses) |clause| try self.visitExpr(clause.expr);
                try self.visitBody(function_decl.body);
            },
            .Field => |field_decl| {
                if (field_decl.value) |value| try self.visitExpr(value);
            },
            .Constant => |constant_decl| try self.visitExpr(constant_decl.value),
            .Trait => |trait_decl| {
                for (trait_decl.methods) |method| {
                    try self.appendSelfNameRange(method.range, method.name);
                    for (method.parameters) |parameter| try self.appendPatternDeclaration(parameter.pattern);
                    for (method.clauses) |clause| try self.visitExpr(clause.expr);
                }
                if (trait_decl.ghost_block) |ghost_id| try self.visitItem(ghost_id);
            },
            .GhostBlock => |ghost_block| try self.visitBody(ghost_block.body),
            .Struct => |struct_decl| {
                for (struct_decl.fields) |field| try self.appendSelfNameRange(field.range, field.name);
            },
            .Bitfield => |bitfield_decl| {
                for (bitfield_decl.fields) |field| try self.appendSelfNameRange(field.range, field.name);
            },
            .Enum => |enum_decl| {
                for (enum_decl.variants) |variant| try self.appendSelfNameRange(variant.range, variant.name);
            },
            .LogDecl => |log_decl| {
                for (log_decl.fields) |field| try self.appendSelfNameRange(field.range, field.name);
            },
            .ErrorDecl => |error_decl| {
                for (error_decl.parameters) |parameter| try self.appendPatternDeclaration(parameter.pattern);
            },
            .Impl => |impl_decl| {
                for (impl_decl.methods) |method_id| try self.visitItem(method_id);
            },
            else => {},
        }
    }

    fn visitBody(self: *DefinitionCollector, body_id: compiler.ast.BodyId) anyerror!void {
        const body = self.resolver.analysis.astFile().body(body_id).*;
        for (body.statements) |stmt_id| try self.visitStmt(stmt_id);
    }

    fn visitStmt(self: *DefinitionCollector, stmt_id: compiler.ast.StmtId) anyerror!void {
        switch (self.resolver.analysis.astFile().statement(stmt_id).*) {
            .VariableDecl => |decl| {
                if (decl.value) |expr_id| try self.visitExpr(expr_id);
                try self.appendPatternDeclaration(decl.pattern);
            },
            .Return => |ret| if (ret.value) |expr_id| try self.visitExpr(expr_id),
            .If => |if_stmt| {
                try self.visitExpr(if_stmt.condition);
                try self.visitBody(if_stmt.then_body);
                if (if_stmt.else_body) |else_body| try self.visitBody(else_body);
            },
            .While => |while_stmt| {
                try self.visitExpr(while_stmt.condition);
                for (while_stmt.invariants) |expr_id| try self.visitExpr(expr_id);
                try self.visitBody(while_stmt.body);
            },
            .For => |for_stmt| {
                try self.visitExpr(for_stmt.iterable);
                try self.appendPatternDeclaration(for_stmt.item_pattern);
                if (for_stmt.index_pattern) |index_pattern| try self.appendPatternDeclaration(index_pattern);
                for (for_stmt.invariants) |expr_id| try self.visitExpr(expr_id);
                try self.visitBody(for_stmt.body);
            },
            .Switch => |switch_stmt| {
                try self.visitExpr(switch_stmt.condition);
                for (switch_stmt.arms) |arm| {
                    try self.visitSwitchPattern(arm.pattern);
                    try self.visitBody(arm.body);
                }
                if (switch_stmt.else_body) |else_body| try self.visitBody(else_body);
            },
            .Try => |try_stmt| {
                try self.visitBody(try_stmt.try_body);
                if (try_stmt.catch_clause) |catch_clause| {
                    if (catch_clause.error_pattern) |pattern_id| try self.appendPatternDeclaration(pattern_id);
                    try self.visitBody(catch_clause.body);
                }
            },
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| try self.visitExpr(arg);
            },
            .Lock => |lock_stmt| try self.visitExpr(lock_stmt.path),
            .Unlock => |unlock_stmt| try self.visitExpr(unlock_stmt.path),
            .Assert => |assert_stmt| try self.visitExpr(assert_stmt.condition),
            .Assume => |assume_stmt| try self.visitExpr(assume_stmt.condition),
            .Break => |jump| if (jump.value) |expr_id| try self.visitExpr(expr_id),
            .Continue => |jump| if (jump.value) |expr_id| try self.visitExpr(expr_id),
            .Assign => |assign| {
                try self.appendPatternDeclaration(assign.target);
                try self.visitPatternUses(assign.target);
                try self.visitExpr(assign.value);
            },
            .Expr => |expr_stmt| try self.visitExpr(expr_stmt.expr),
            .Block => |block| try self.visitBody(block.body),
            .LabeledBlock => |block| try self.visitBody(block.body),
            else => {},
        }
    }

    fn visitPatternUses(self: *DefinitionCollector, pattern_id: compiler.ast.PatternId) anyerror!void {
        switch (self.resolver.analysis.astFile().pattern(pattern_id).*) {
            .Field => |field| try self.visitPatternUses(field.base),
            .Index => |index| {
                try self.visitPatternUses(index.base);
                try self.visitExpr(index.index);
            },
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| {
                    try self.appendSelfNameRange(field.range, field.name);
                    try self.visitPatternUses(field.binding);
                }
            },
            else => {},
        }
    }

    fn visitExpr(self: *DefinitionCollector, expr_id: compiler.ast.ExprId) anyerror!void {
        const expr = self.resolver.analysis.astFile().expression(expr_id).*;
        switch (expr) {
            .Name => |name| try self.appendExprName(expr_id, name.range, name.name),
            .Unary => |unary| try self.visitExpr(unary.operand),
            .Binary => |binary| {
                try self.visitExpr(binary.lhs);
                try self.visitExpr(binary.rhs);
            },
            .Tuple => |tuple| {
                for (tuple.elements) |element| try self.visitExpr(element);
            },
            .ArrayLiteral => |array| {
                for (array.elements) |element| try self.visitExpr(element);
            },
            .StructLiteral => |struct_literal| {
                for (struct_literal.fields) |field| {
                    try self.appendSelfNameRange(field.range, field.name);
                    try self.visitExpr(field.value);
                }
            },
            .Switch => |switch_expr| {
                try self.visitExpr(switch_expr.condition);
                for (switch_expr.arms) |arm| {
                    try self.visitSwitchPattern(arm.pattern);
                    try self.visitExpr(arm.value);
                }
                if (switch_expr.else_expr) |else_expr| try self.visitExpr(else_expr);
            },
            .Comptime => |comptime_expr| try self.visitBody(comptime_expr.body),
            .ErrorReturn => |error_return| {
                for (error_return.args) |arg| try self.visitExpr(arg);
            },
            .Call => |call| {
                try self.visitExpr(call.callee);
                for (call.args) |arg| try self.visitExpr(arg);
            },
            .Builtin => |builtin| {
                for (builtin.args) |arg| try self.visitExpr(arg);
            },
            .Field => |field| try self.visitExpr(field.base),
            .Index => |index| {
                try self.visitExpr(index.base);
                try self.visitExpr(index.index);
            },
            .Group => |group| try self.visitExpr(group.expr),
            .Old => |old| try self.visitExpr(old.expr),
            .Quantified => |quantified| {
                try self.appendPatternDeclaration(quantified.pattern);
                if (quantified.condition) |condition| try self.visitExpr(condition);
                try self.visitExpr(quantified.body);
            },
            else => {},
        }
    }

    fn visitSwitchPattern(self: *DefinitionCollector, pattern: compiler.ast.SwitchPattern) anyerror!void {
        switch (pattern) {
            .Expr => |expr_id| try self.visitExpr(expr_id),
            .Range => |range_pattern| {
                try self.visitExpr(range_pattern.start);
                try self.visitExpr(range_pattern.end);
            },
            .NamedError => |named_error| {
                try self.visitExpr(named_error.callee);
                for (named_error.bindings) |pattern_id| try self.appendPatternDeclaration(pattern_id);
            },
            .Or => |or_pattern| {
                for (or_pattern.alternatives) |alternative| try self.visitSwitchPattern(alternative);
            },
            .Ok, .Err => |pattern_id| try self.appendPatternDeclaration(pattern_id),
            .Else => {},
        }
    }

    fn appendItemDeclaration(self: *DefinitionCollector, item_id: compiler.ast.ItemId) !void {
        const pair = self.resolver.itemSelectionPair(item_id) orelse return;
        try self.appendResolvedName(pair.name, pair.range, pair.range);
    }

    fn appendPatternDeclaration(self: *DefinitionCollector, pattern_id: compiler.ast.PatternId) !void {
        const pair = self.resolver.patternSelectionPair(pattern_id) orelse return;
        const name = pair.name orelse return;
        try self.appendResolvedName(name, pair.range, pair.range);
    }

    fn appendSelfNameRange(self: *DefinitionCollector, range: compiler.TextRange, name: []const u8) !void {
        const selection = self.resolver.analysis.selectionRange(range, name);
        try self.appendResolvedName(name, selection, selection);
    }

    fn appendExprName(self: *DefinitionCollector, expr_id: compiler.ast.ExprId, range: compiler.TextRange, name: []const u8) !void {
        const selection = self.resolver.analysis.selectionRange(range, name);
        const resolved = if (self.resolver.importAliasItemByName(name)) |item_id|
            try self.resolver.definitionForItem(item_id)
        else blk: {
            const binding = self.resolver.analysis.resolutionView().expr_bindings[expr_id.index()] orelse break :blk null;
            break :blk try self.resolver.definitionForBinding(binding);
        };
        const definition = resolved orelse return;
        try self.appendResolvedName(name, selection, definition.range);
    }

    fn appendResolvedName(self: *DefinitionCollector, name: []const u8, range: frontend.Range, definition_range: frontend.Range) !void {
        try self.names.append(self.allocator, .{
            .name = name,
            .range = range,
            .definition_range = definition_range,
        });
    }
};

/// Collect all same-file identifier definitions in one AST/name-resolution pass.
pub fn collectDefinitionsCached(allocator: Allocator, analysis: *const Analysis) ![]ResolvedName {
    var collector = DefinitionCollector.init(allocator, analysis);
    return collector.run();
}

/// Resolve definition using a pre-built analysis (avoids re-parsing).
pub fn definitionAtCached(analysis: *Analysis, source: []const u8, position: frontend.Position) ?Definition {
    return definitionAtAnalysis(analysis, source, position, null) catch null;
}

pub fn definitionAtCachedCrossFile(analysis: *Analysis, source: []const u8, position: frontend.Position, cross_file: CrossFileContext) !?Definition {
    return definitionAtAnalysis(analysis, source, position, cross_file);
}

fn definitionAtAnalysis(analysis: *Analysis, source: []const u8, position: frontend.Position, cross_file: ?CrossFileContext) !?Definition {
    _ = source;

    var resolver = Resolver{
        .analysis = analysis,
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
