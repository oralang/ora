const std = @import("std");
const ids = @import("ids.zig");
const nodes = @import("nodes.zig");

const AstFileId = ids.AstFileId;
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

pub const AstFile = struct {
    arena: std.heap.ArenaAllocator,
    file_id: AstFileId,
    root_items: []ItemId,
    items: []Item,
    bodies: []Body,
    statements: []Stmt,
    expressions: []Expr,
    type_exprs: []TypeExpr,
    patterns: []Pattern,

    pub fn deinit(self: *AstFile) void {
        self.arena.deinit();
    }

    pub fn allocator(self: *AstFile) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn item(self: *const AstFile, id: ItemId) *const Item {
        return &self.items[id.index()];
    }

    pub fn body(self: *const AstFile, id: BodyId) *const Body {
        return &self.bodies[id.index()];
    }

    pub fn statement(self: *const AstFile, id: StmtId) *const Stmt {
        return &self.statements[id.index()];
    }

    pub fn expression(self: *const AstFile, id: ExprId) *const Expr {
        return &self.expressions[id.index()];
    }

    pub fn typeExpr(self: *const AstFile, id: TypeExprId) *const TypeExpr {
        return &self.type_exprs[id.index()];
    }

    pub fn pattern(self: *const AstFile, id: PatternId) *const Pattern {
        return &self.patterns[id.index()];
    }
};
