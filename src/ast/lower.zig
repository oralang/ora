const std = @import("std");
const diagnostics = @import("../diagnostics/mod.zig");
const source = @import("../source/mod.zig");
const syntax = @import("../syntax/mod.zig");
const ids = @import("ids.zig");
const nodes = @import("nodes.zig");
const ast_file = @import("file.zig");
const ast_builder = @import("builder.zig");

const ItemId = ids.ItemId;
const BodyId = ids.BodyId;
const StmtId = ids.StmtId;
const ExprId = ids.ExprId;
const TypeExprId = ids.TypeExprId;
const PatternId = ids.PatternId;
const Item = nodes.Item;
const Body = nodes.Body;
const Stmt = nodes.Stmt;
const Expr = nodes.Expr;
const TypeExpr = nodes.TypeExpr;
const Pattern = nodes.Pattern;
const AstFile = ast_file.AstFile;
const Builder = ast_builder.Builder;

pub const LowerResult = struct {
    file: AstFile,
    diagnostics: diagnostics.DiagnosticList,

    pub fn deinit(self: *LowerResult) void {
        self.file.deinit();
        self.diagnostics.deinit();
    }
};

pub fn lower(allocator: std.mem.Allocator, tree: *const syntax.SyntaxTree) !LowerResult {
    var result = LowerResult{
        .file = AstFile{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .file_id = tree.file_id,
            .root_items = &[_]ItemId{},
            .items = &[_]Item{},
            .bodies = &[_]Body{},
            .statements = &[_]Stmt{},
            .expressions = &[_]Expr{},
            .type_exprs = &[_]TypeExpr{},
            .patterns = &[_]Pattern{},
        },
        .diagnostics = diagnostics.DiagnosticList.init(allocator),
    };
    errdefer result.diagnostics.deinit();
    errdefer result.file.deinit();

    var builder = Builder.init(&result.file, tree, &result.diagnostics);
    try builder.parseFile();
    try builder.finish();
    try validate(&result.file, &result.diagnostics);
    return result;
}

pub fn debugDump(allocator: std.mem.Allocator, file: *const AstFile) ![]u8 {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);

    const writer = buffer.writer(allocator);
    try writer.print("AstFile(file={d})\n", .{@intFromEnum(file.file_id)});
    for (file.root_items, 0..) |item_id, index| {
        try writer.print("  [{d}] {s}\n", .{ index, @tagName(std.meta.activeTag(file.item(item_id).*)) });
    }
    return buffer.toOwnedSlice(allocator);
}

pub fn validate(file: *const AstFile, diags: *diagnostics.DiagnosticList) !void {
    var validator = Validator{
        .file = file,
        .diags = diags,
    };
    try validator.run();
}

const Validator = struct {
    file: *const AstFile,
    diags: *diagnostics.DiagnosticList,

    fn run(self: *Validator) !void {
        for (self.file.root_items) |item_id| {
            _ = try self.expectItem(item_id, source.TextRange.empty(0), "root item references invalid item id");
        }
        try self.validateItemScope(self.file.root_items, "root scope");

        for (self.file.items, 0..) |item, index| {
            try self.validateItem(ItemId.fromIndex(index), item);
        }
        for (self.file.bodies, 0..) |body, index| {
            try self.validateBody(BodyId.fromIndex(index), body);
        }
        for (self.file.statements, 0..) |statement, index| {
            try self.validateStatement(StmtId.fromIndex(index), statement);
        }
        for (self.file.expressions, 0..) |expr, index| {
            try self.validateExpr(ExprId.fromIndex(index), expr);
        }
        for (self.file.type_exprs, 0..) |type_expr, index| {
            try self.validateTypeExpr(TypeExprId.fromIndex(index), type_expr);
        }
        for (self.file.patterns, 0..) |pattern, index| {
            try self.validatePattern(PatternId.fromIndex(index), pattern);
        }
    }

    fn validateItem(self: *Validator, item_id: ItemId, item: Item) !void {
        const item_range = source.rangeOf(item);
        switch (item) {
            .Contract => |contract| {
                for (contract.members) |member_id| {
                    _ = try self.expectItem(member_id, item_range, "contract references invalid member item id");
                }
                for (contract.invariants) |expr_id| {
                    _ = try self.expectExpr(expr_id, item_range, "contract references invalid invariant expression id");
                }
                try self.validateItemScope(contract.members, "contract scope");
            },
            .Function => |function| {
                if (function.return_type) |type_id| {
                    _ = try self.expectType(type_id, item_range, "function references invalid return type id");
                }
                _ = try self.expectBody(function.body, item_range, "function references invalid body id");
                if (function.parent_contract) |parent_id| {
                    _ = try self.expectItem(parent_id, item_range, "function references invalid parent contract id");
                }
                try self.validateParameters(function.parameters, "function parameter");
                for (function.clauses) |clause| {
                    _ = try self.expectExpr(clause.expr, clause.range, "function clause references invalid expression id");
                }
            },
            .Struct => |struct_item| try self.validateStructFields(struct_item.fields, "struct field"),
            .Bitfield => |bitfield| {
                if (bitfield.base_type) |type_id| {
                    _ = try self.expectType(type_id, item_range, "bitfield references invalid base type id");
                }
                try self.validateBitfieldFields(bitfield.fields);
            },
            .Enum => |enum_item| try self.validateNames(EnumVariantAdapter.init(enum_item.variants), "duplicate enum variant name '{s}'"),
            .Trait => |trait_item| {
                var seen: std.StringHashMap(source.TextRange) = .init(self.diags.allocator);
                defer seen.deinit();
                for (trait_item.methods) |method| {
                    if (seen.get(method.name)) |first_range| {
                        try self.emitError(method.range, "duplicate trait method '{s}'", .{
                            method.name,
                        });
                        _ = first_range;
                    } else {
                        try seen.put(method.name, method.range);
                    }
                    if (method.return_type) |type_id| {
                        _ = try self.expectType(type_id, method.range, "trait method references invalid return type id");
                    }
                    try self.validateParameters(method.parameters, "trait method parameter");
                    for (method.clauses) |clause| {
                        _ = try self.expectExpr(clause.expr, clause.range, "trait method clause references invalid expression id");
                    }
                }
            },
            .Impl => |impl_item| {
                for (impl_item.methods) |method_id| {
                    _ = try self.expectItem(method_id, item_range, "impl references invalid method item id");
                }
                try self.validateItemScope(impl_item.methods, "impl scope");
            },
            .TypeAlias => |type_alias| {
                _ = try self.expectType(type_alias.target_type, item_range, "type alias references invalid target type id");
                try self.validateParameters(type_alias.template_parameters, "type alias parameter");
            },
            .LogDecl => |log_decl| try self.validateLogFields(log_decl.fields),
            .ErrorDecl => |error_decl| try self.validateParameters(error_decl.parameters, "error parameter"),
            .GhostBlock => |ghost_block| {
                _ = try self.expectBody(ghost_block.body, item_range, "ghost block references invalid body id");
            },
            .Field => |field| {
                if (field.type_expr) |type_id| {
                    _ = try self.expectType(type_id, item_range, "field references invalid type id");
                }
                if (field.value) |expr_id| {
                    _ = try self.expectExpr(expr_id, item_range, "field references invalid initializer expression id");
                }
            },
            .Constant => |constant| {
                if (constant.type_expr) |type_id| {
                    _ = try self.expectType(type_id, item_range, "constant references invalid type id");
                }
                _ = try self.expectExpr(constant.value, item_range, "constant references invalid value expression id");
            },
            .Import, .Error => {},
        }
        _ = item_id;
    }

    fn validateBody(self: *Validator, _: BodyId, body: Body) !void {
        for (body.statements) |statement_id| {
            _ = try self.expectStmt(statement_id, body.range, "body references invalid statement id");
        }
    }

    fn validateStatement(self: *Validator, _: StmtId, statement: Stmt) !void {
        const stmt_range = source.rangeOf(statement);
        switch (statement) {
            .VariableDecl => |decl| {
                _ = try self.expectPattern(decl.pattern, stmt_range, "variable declaration references invalid pattern id");
                if (decl.type_expr) |type_id| _ = try self.expectType(type_id, stmt_range, "variable declaration references invalid type id");
                if (decl.value) |expr_id| _ = try self.expectExpr(expr_id, stmt_range, "variable declaration references invalid initializer expression id");
            },
            .Return => |ret| {
                if (ret.value) |expr_id| _ = try self.expectExpr(expr_id, stmt_range, "return references invalid expression id");
            },
            .If => |if_stmt| {
                _ = try self.expectExpr(if_stmt.condition, stmt_range, "if statement references invalid condition expression id");
                _ = try self.expectBody(if_stmt.then_body, stmt_range, "if statement references invalid then body id");
                if (if_stmt.else_body) |body_id| _ = try self.expectBody(body_id, stmt_range, "if statement references invalid else body id");
            },
            .While => |while_stmt| {
                _ = try self.expectExpr(while_stmt.condition, stmt_range, "while statement references invalid condition expression id");
                for (while_stmt.invariants) |expr_id| _ = try self.expectExpr(expr_id, stmt_range, "while statement references invalid invariant expression id");
                _ = try self.expectBody(while_stmt.body, stmt_range, "while statement references invalid body id");
            },
            .For => |for_stmt| {
                _ = try self.expectExpr(for_stmt.iterable, stmt_range, "for statement references invalid iterable expression id");
                _ = try self.expectPattern(for_stmt.item_pattern, stmt_range, "for statement references invalid item pattern id");
                if (for_stmt.index_pattern) |pattern_id| _ = try self.expectPattern(pattern_id, stmt_range, "for statement references invalid index pattern id");
                for (for_stmt.invariants) |expr_id| _ = try self.expectExpr(expr_id, stmt_range, "for statement references invalid invariant expression id");
                _ = try self.expectBody(for_stmt.body, stmt_range, "for statement references invalid body id");
            },
            .Switch => |switch_stmt| {
                _ = try self.expectExpr(switch_stmt.condition, stmt_range, "switch statement references invalid condition expression id");
                for (switch_stmt.arms) |arm| {
                    try self.validateSwitchPattern(arm.pattern, arm.range);
                    _ = try self.expectBody(arm.body, arm.range, "switch arm references invalid body id");
                }
                if (switch_stmt.else_body) |body_id| _ = try self.expectBody(body_id, stmt_range, "switch statement references invalid else body id");
            },
            .Break, .Continue => |jump| {
                if (jump.value) |expr_id| _ = try self.expectExpr(expr_id, stmt_range, "jump statement references invalid expression id");
            },
            .Try => |try_stmt| {
                _ = try self.expectBody(try_stmt.try_body, stmt_range, "try statement references invalid try body id");
                if (try_stmt.catch_clause) |catch_clause| {
                    if (catch_clause.error_pattern) |pattern_id| _ = try self.expectPattern(pattern_id, catch_clause.range, "catch clause references invalid error pattern id");
                    _ = try self.expectBody(catch_clause.body, catch_clause.range, "catch clause references invalid body id");
                }
            },
            .Log => |log_stmt| {
                for (log_stmt.args) |expr_id| _ = try self.expectExpr(expr_id, stmt_range, "log statement references invalid argument expression id");
            },
            .Lock => |lock_stmt| _ = try self.expectExpr(lock_stmt.path, stmt_range, "lock statement references invalid path expression id"),
            .Unlock => |unlock_stmt| _ = try self.expectExpr(unlock_stmt.path, stmt_range, "unlock statement references invalid path expression id"),
            .Assert => |assert_stmt| _ = try self.expectExpr(assert_stmt.condition, stmt_range, "assert statement references invalid condition expression id"),
            .Assume => |assume_stmt| _ = try self.expectExpr(assume_stmt.condition, stmt_range, "assume statement references invalid condition expression id"),
            .Assign => |assign| {
                _ = try self.expectPattern(assign.target, stmt_range, "assignment references invalid target pattern id");
                _ = try self.expectExpr(assign.value, stmt_range, "assignment references invalid value expression id");
            },
            .Expr => |expr_stmt| _ = try self.expectExpr(expr_stmt.expr, stmt_range, "expression statement references invalid expression id"),
            .Block => |block| _ = try self.expectBody(block.body, stmt_range, "block statement references invalid body id"),
            .LabeledBlock => |block| _ = try self.expectBody(block.body, stmt_range, "labeled block references invalid body id"),
            .Havoc, .Error => {},
        }
    }

    fn validateExpr(self: *Validator, _: ExprId, expr: Expr) !void {
        const expr_range = source.rangeOf(expr);
        switch (expr) {
            .TypeValue => {},
            .Tuple => |tuple| {
                for (tuple.elements) |expr_id| _ = try self.expectExpr(expr_id, expr_range, "tuple expression references invalid element id");
            },
            .ArrayLiteral => |array| {
                for (array.elements) |expr_id| _ = try self.expectExpr(expr_id, expr_range, "array literal references invalid element id");
            },
            .StructLiteral => |struct_literal| {
                if (struct_literal.type_expr) |type_id| {
                    _ = try self.expectType(type_id, expr_range, "struct literal references invalid type id");
                }
                for (struct_literal.fields) |field| _ = try self.expectExpr(field.value, field.range, "struct literal field references invalid expression id");
            },
            .Switch => |switch_expr| {
                _ = try self.expectExpr(switch_expr.condition, expr_range, "switch expression references invalid condition expression id");
                for (switch_expr.arms) |arm| {
                    try self.validateSwitchPattern(arm.pattern, arm.range);
                    _ = try self.expectExpr(arm.value, arm.range, "switch expression arm references invalid value expression id");
                }
                if (switch_expr.else_expr) |expr_id| _ = try self.expectExpr(expr_id, expr_range, "switch expression references invalid else expression id");
            },
            .Comptime => |comptime_expr| _ = try self.expectBody(comptime_expr.body, expr_range, "comptime expression references invalid body id"),
            .ExternalProxy => |proxy| {
                _ = try self.expectExpr(proxy.address_expr, expr_range, "external proxy references invalid address expression id");
                _ = try self.expectExpr(proxy.gas_expr, expr_range, "external proxy references invalid gas expression id");
            },
            .ErrorReturn => |error_return| {
                for (error_return.args) |expr_id| _ = try self.expectExpr(expr_id, expr_range, "error return references invalid argument expression id");
            },
            .Unary => |unary| _ = try self.expectExpr(unary.operand, expr_range, "unary expression references invalid operand id"),
            .Binary => |binary| {
                _ = try self.expectExpr(binary.lhs, expr_range, "binary expression references invalid lhs expression id");
                _ = try self.expectExpr(binary.rhs, expr_range, "binary expression references invalid rhs expression id");
            },
            .Call => |call| {
                _ = try self.expectExpr(call.callee, expr_range, "call expression references invalid callee expression id");
                for (call.args) |expr_id| _ = try self.expectExpr(expr_id, expr_range, "call expression references invalid argument expression id");
            },
            .Builtin => |builtin| {
                if (builtin.type_arg) |type_id| _ = try self.expectType(type_id, expr_range, "builtin expression references invalid type argument id");
                for (builtin.args) |expr_id| _ = try self.expectExpr(expr_id, expr_range, "builtin expression references invalid argument expression id");
            },
            .Field => |field| _ = try self.expectExpr(field.base, expr_range, "field expression references invalid base expression id"),
            .Index => |index| {
                _ = try self.expectExpr(index.base, expr_range, "index expression references invalid base expression id");
                _ = try self.expectExpr(index.index, expr_range, "index expression references invalid index expression id");
            },
            .Group => |group| _ = try self.expectExpr(group.expr, expr_range, "group expression references invalid inner expression id"),
            .Old => |old| _ = try self.expectExpr(old.expr, expr_range, "old expression references invalid inner expression id"),
            .Quantified => |quantified| {
                _ = try self.expectPattern(quantified.pattern, expr_range, "quantified expression references invalid pattern id");
                _ = try self.expectType(quantified.type_expr, expr_range, "quantified expression references invalid type id");
                if (quantified.condition) |condition| _ = try self.expectExpr(condition, expr_range, "quantified expression references invalid condition expression id");
                _ = try self.expectExpr(quantified.body, expr_range, "quantified expression references invalid body expression id");
            },
            .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Name, .Result, .Error => {},
        }
    }

    fn validateTypeExpr(self: *Validator, _: TypeExprId, type_expr: TypeExpr) !void {
        const type_range = source.rangeOf(type_expr);
        switch (type_expr) {
            .Generic => |generic| {
                for (generic.args) |arg| switch (arg) {
                    .Type => |type_id| _ = try self.expectType(type_id, type_range, "generic type references invalid type argument id"),
                    .Integer => {},
                };
            },
            .Tuple => |tuple| {
                for (tuple.elements) |type_id| _ = try self.expectType(type_id, type_range, "tuple type references invalid element type id");
            },
            .AnonymousStruct => |struct_type| {
                try self.validateNames(AnonymousStructFieldAdapter.init(struct_type.fields), "duplicate anonymous struct field '{s}'");
                for (struct_type.fields) |field| {
                    _ = try self.expectType(field.type_expr, field.range, "anonymous struct type references invalid field type id");
                }
            },
            .Array => |array| _ = try self.expectType(array.element, type_range, "array type references invalid element type id"),
            .Slice => |slice| _ = try self.expectType(slice.element, type_range, "slice type references invalid element type id"),
            .ErrorUnion => |union_type| {
                _ = try self.expectType(union_type.payload, type_range, "error union references invalid payload type id");
                for (union_type.errors) |type_id| _ = try self.expectType(type_id, type_range, "error union references invalid error type id");
            },
            .Path, .Error => {},
        }
    }

    fn validatePattern(self: *Validator, _: PatternId, pattern: Pattern) !void {
        const pattern_range = source.rangeOf(pattern);
        switch (pattern) {
            .Field => |field| _ = try self.expectPattern(field.base, pattern_range, "field pattern references invalid base pattern id"),
            .Index => |index| {
                _ = try self.expectPattern(index.base, pattern_range, "index pattern references invalid base pattern id");
                _ = try self.expectExpr(index.index, pattern_range, "index pattern references invalid index expression id");
            },
            .StructDestructure => |destructure| {
                try self.validateNames(StructDestructureFieldAdapter.init(destructure.fields), "duplicate destructuring field name '{s}'");
                for (destructure.fields) |field| {
                    _ = try self.expectPattern(field.binding, field.range, "struct destructuring field references invalid binding pattern id");
                }
            },
            .Name, .Error => {},
        }
    }

    fn validateParameters(self: *Validator, parameters: []const nodes.Parameter, label: []const u8) !void {
        var seen: std.StringHashMap(source.TextRange) = .init(self.diags.allocator);
        defer seen.deinit();
        for (parameters) |parameter| {
            const pattern_ptr = try self.expectPattern(parameter.pattern, parameter.range, "parameter references invalid pattern id") orelse continue;
            _ = try self.expectType(parameter.type_expr, parameter.range, "parameter references invalid type id");

            const pattern = pattern_ptr.*;
            if (pattern != .Name) continue;
            const name = pattern.Name.name;
            if (name.len == 0) continue;
            if (seen.get(name)) |_| {
                try self.emitError(parameter.range, "{s} name '{s}' is declared more than once", .{ label, name });
            } else {
                try seen.put(name, parameter.range);
            }
        }
    }

    const AnonymousStructFieldAdapter = struct {
        slice: []const nodes.AnonymousStructFieldType,

        fn init(fields: []const nodes.AnonymousStructFieldType) AnonymousStructFieldAdapter {
            return .{ .slice = fields };
        }

        pub fn name(self: AnonymousStructFieldAdapter, entry: nodes.AnonymousStructFieldType) []const u8 {
            _ = self;
            return entry.name;
        }

        pub fn range(self: AnonymousStructFieldAdapter, entry: nodes.AnonymousStructFieldType) source.TextRange {
            _ = self;
            return entry.range;
        }
    };

    fn validateStructFields(self: *Validator, fields: []const nodes.StructField, noun: []const u8) !void {
        var seen: std.StringHashMap(source.TextRange) = .init(self.diags.allocator);
        defer seen.deinit();
        for (fields) |field| {
            _ = try self.expectType(field.type_expr, field.range, "field references invalid type id");
            if (field.name.len == 0) continue;
            if (seen.get(field.name)) |_| {
                try self.emitError(field.range, "duplicate {s} name '{s}'", .{ noun, field.name });
            } else {
                try seen.put(field.name, field.range);
            }
        }
    }

    fn validateBitfieldFields(self: *Validator, fields: []const nodes.BitfieldField) !void {
        var seen: std.StringHashMap(source.TextRange) = .init(self.diags.allocator);
        defer seen.deinit();
        for (fields) |field| {
            _ = try self.expectType(field.type_expr, field.range, "bitfield field references invalid type id");
            if (field.name.len == 0) continue;
            if (seen.get(field.name)) |_| {
                try self.emitError(field.range, "duplicate bitfield field name '{s}'", .{field.name});
            } else {
                try seen.put(field.name, field.range);
            }
        }
    }

    fn validateLogFields(self: *Validator, fields: []const nodes.LogField) !void {
        var seen: std.StringHashMap(source.TextRange) = .init(self.diags.allocator);
        defer seen.deinit();
        for (fields) |field| {
            _ = try self.expectType(field.type_expr, field.range, "log field references invalid type id");
            if (field.name.len == 0) continue;
            if (seen.get(field.name)) |_| {
                try self.emitError(field.range, "duplicate log field name '{s}'", .{field.name});
            } else {
                try seen.put(field.name, field.range);
            }
        }
    }

    fn validateItemScope(self: *Validator, item_ids: []const ItemId, scope_name: []const u8) !void {
        var seen: std.StringHashMap(source.TextRange) = .init(self.diags.allocator);
        defer seen.deinit();
        for (item_ids) |item_id| {
            const item = self.expectItem(item_id, source.TextRange.empty(0), "scope references invalid item id") catch null orelse continue;
            const name = itemName(item.*) orelse continue;
            if (name.len == 0) continue;
            if (seen.get(name)) |_| {
                try self.emitError(source.rangeOf(item.*), "duplicate item name '{s}' in {s}", .{ name, scope_name });
            } else {
                try seen.put(name, source.rangeOf(item.*));
            }
        }
    }

    fn validateNames(self: *Validator, adapter: anytype, comptime fmt: []const u8) !void {
        var seen: std.StringHashMap(source.TextRange) = .init(self.diags.allocator);
        defer seen.deinit();
        for (adapter.slice) |entry| {
            const name = adapter.name(entry);
            if (name.len == 0) continue;
            if (seen.get(name)) |_| {
                try self.emitError(adapter.range(entry), fmt, .{name});
            } else {
                try seen.put(name, adapter.range(entry));
            }
        }
    }

    fn validateSwitchPattern(self: *Validator, pattern: nodes.SwitchPattern, owner_range: source.TextRange) !void {
        switch (pattern) {
            .Expr => |expr_id| _ = try self.expectExpr(expr_id, owner_range, "switch pattern references invalid expression id"),
            .Range => |range_pattern| {
                _ = try self.expectExpr(range_pattern.start, range_pattern.range, "switch range pattern references invalid start expression id");
                _ = try self.expectExpr(range_pattern.end, range_pattern.range, "switch range pattern references invalid end expression id");
            },
            .Ok, .Err => |pattern_id| _ = try self.expectPattern(pattern_id, owner_range, "switch match pattern references invalid binding pattern id"),
            .Else => {},
        }
    }

    fn expectItem(self: *Validator, item_id: ItemId, range: source.TextRange, message: []const u8) !?*const Item {
        if (item_id.index() >= self.file.items.len) {
            try self.simpleError(range, message);
            return null;
        }
        return self.file.item(item_id);
    }

    fn expectBody(self: *Validator, body_id: BodyId, range: source.TextRange, message: []const u8) !?*const Body {
        if (body_id.index() >= self.file.bodies.len) {
            try self.simpleError(range, message);
            return null;
        }
        return self.file.body(body_id);
    }

    fn expectStmt(self: *Validator, stmt_id: StmtId, range: source.TextRange, message: []const u8) !?*const Stmt {
        if (stmt_id.index() >= self.file.statements.len) {
            try self.simpleError(range, message);
            return null;
        }
        return self.file.statement(stmt_id);
    }

    fn expectExpr(self: *Validator, expr_id: ExprId, range: source.TextRange, message: []const u8) !?*const Expr {
        if (expr_id.index() >= self.file.expressions.len) {
            try self.simpleError(range, message);
            return null;
        }
        return self.file.expression(expr_id);
    }

    fn expectType(self: *Validator, type_id: TypeExprId, range: source.TextRange, message: []const u8) !?*const TypeExpr {
        if (type_id.index() >= self.file.type_exprs.len) {
            try self.simpleError(range, message);
            return null;
        }
        return self.file.typeExpr(type_id);
    }

    fn expectPattern(self: *Validator, pattern_id: PatternId, range: source.TextRange, message: []const u8) !?*const Pattern {
        if (pattern_id.index() >= self.file.patterns.len) {
            try self.simpleError(range, message);
            return null;
        }
        return self.file.pattern(pattern_id);
    }

    fn simpleError(self: *Validator, range: source.TextRange, message: []const u8) !void {
        try self.diags.appendError(message, .{
            .file_id = self.file.file_id,
            .range = range,
        });
    }

    fn emitError(self: *Validator, range: source.TextRange, comptime fmt: []const u8, args: anytype) !void {
        var buffer: [256]u8 = undefined;
        const message = try std.fmt.bufPrint(&buffer, fmt, args);
        try self.simpleError(range, message);
    }
};

fn itemName(item: Item) ?[]const u8 {
    return switch (item) {
        .Contract => |contract| contract.name,
        .Function => |function| function.name,
        .Struct => |struct_item| struct_item.name,
        .Bitfield => |bitfield| bitfield.name,
        .Enum => |enum_item| enum_item.name,
        .TypeAlias => |type_alias| type_alias.name,
        .LogDecl => |log_decl| log_decl.name,
        .ErrorDecl => |error_decl| error_decl.name,
        .Field => |field| field.name,
        .Constant => |constant| constant.name,
        else => null,
    };
}

const EnumVariantAdapter = struct {
    slice: []const nodes.EnumVariant,

    fn init(slice: []const nodes.EnumVariant) EnumVariantAdapter {
        return .{ .slice = slice };
    }

    fn name(_: EnumVariantAdapter, entry: nodes.EnumVariant) []const u8 {
        return entry.name;
    }

    fn range(_: EnumVariantAdapter, entry: nodes.EnumVariant) source.TextRange {
        return entry.range;
    }
};

const StructDestructureFieldAdapter = struct {
    slice: []const nodes.StructDestructureField,

    fn init(slice: []const nodes.StructDestructureField) StructDestructureFieldAdapter {
        return .{ .slice = slice };
    }

    fn name(_: StructDestructureFieldAdapter, entry: nodes.StructDestructureField) []const u8 {
        return entry.name;
    }

    fn range(_: StructDestructureFieldAdapter, entry: nodes.StructDestructureField) source.TextRange {
        return entry.range;
    }
};
