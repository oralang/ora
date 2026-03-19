const source = @import("../source/mod.zig");
const ids = @import("ids.zig");
const nodes = @import("nodes.zig");

const ItemId = ids.ItemId;
const BodyId = ids.BodyId;
const StmtId = ids.StmtId;
const ExprId = ids.ExprId;
const TypeExprId = ids.TypeExprId;
const PatternId = ids.PatternId;
const TypeExpr = nodes.TypeExpr;
const Pattern = nodes.Pattern;
const Expr = nodes.Expr;
const Stmt = nodes.Stmt;
const Body = nodes.Body;
const Item = nodes.Item;

pub fn mixin(Builder: type) type {
    return struct {
        pub fn pushItem(self: *Builder, item: Item) !ItemId {
            const id = ItemId.fromIndex(self.items.items.len);
            try self.items.append(self.allocator, item);
            return id;
        }

        pub fn pushBody(self: *Builder, body: Body) !BodyId {
            const id = BodyId.fromIndex(self.bodies.items.len);
            try self.bodies.append(self.allocator, body);
            return id;
        }

        pub fn pushStmt(self: *Builder, stmt: Stmt) !StmtId {
            const id = StmtId.fromIndex(self.statements.items.len);
            try self.statements.append(self.allocator, stmt);
            return id;
        }

        pub fn pushExpr(self: *Builder, expr: Expr) !ExprId {
            const id = ExprId.fromIndex(self.expressions.items.len);
            try self.expressions.append(self.allocator, expr);
            return id;
        }

        pub fn pushTypeExpr(self: *Builder, expr: TypeExpr) !TypeExprId {
            const id = TypeExprId.fromIndex(self.type_exprs.items.len);
            try self.type_exprs.append(self.allocator, expr);
            return id;
        }

        pub fn pushPattern(self: *Builder, pattern: Pattern) !PatternId {
            const id = PatternId.fromIndex(self.patterns.items.len);
            try self.patterns.append(self.allocator, pattern);
            return id;
        }

        pub fn bodyRef(self: *const Builder, id: BodyId) *const Body {
            return &self.bodies.items[id.index()];
        }

        pub fn exprRef(self: *const Builder, id: ExprId) *const Expr {
            return &self.expressions.items[id.index()];
        }

        pub fn typeExprRef(self: *const Builder, id: TypeExprId) *const TypeExpr {
            return &self.type_exprs.items[id.index()];
        }

        pub fn stmtRef(self: *const Builder, id: StmtId) *const Stmt {
            return &self.statements.items[id.index()];
        }

        pub fn exprRange(self: *const Builder, id: ExprId) source.TextRange {
            return source.rangeOf(exprRef(self, id).*);
        }
    };
}

pub fn stripQuotes(text: []const u8) []const u8 {
    if (text.len >= 2 and ((text[0] == '"' and text[text.len - 1] == '"') or (text[0] == '\'' and text[text.len - 1] == '\''))) {
        return text[1 .. text.len - 1];
    }
    return text;
}
