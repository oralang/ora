const std = @import("std");
const diagnostics = @import("../diagnostics/mod.zig");
const syntax = @import("../syntax/mod.zig");
const ids = @import("ids.zig");
const nodes = @import("nodes.zig");
const ast_file = @import("file.zig");
const support = @import("support.zig");
const syntax_lowering = @import("syntax_lowering.zig");

const ItemId = ids.ItemId;
const TypeExpr = nodes.TypeExpr;
const Pattern = nodes.Pattern;
const Expr = nodes.Expr;
const Stmt = nodes.Stmt;
const Body = nodes.Body;
const Item = nodes.Item;
const AstFile = ast_file.AstFile;

pub const Builder = struct {
    file: *AstFile,
    tree: *const syntax.SyntaxTree,
    diagnostics: *diagnostics.DiagnosticList,
    allocator: std.mem.Allocator,
    root_items: std.ArrayList(ItemId),
    items: std.ArrayList(Item),
    bodies: std.ArrayList(Body),
    statements: std.ArrayList(Stmt),
    expressions: std.ArrayList(Expr),
    type_exprs: std.ArrayList(TypeExpr),
    patterns: std.ArrayList(Pattern),

    const Support = support.mixin(Builder);
    const SyntaxLowering = syntax_lowering.mixin(Builder);
    pub const lowerFileFromSyntax = SyntaxLowering.lowerFileFromSyntax;

    pub fn init(file: *AstFile, tree: *const syntax.SyntaxTree, diags: *diagnostics.DiagnosticList) Builder {
        const arena_allocator = file.arena.allocator();
        return .{
            .file = file,
            .tree = tree,
            .diagnostics = diags,
            .allocator = arena_allocator,
            .root_items = .empty,
            .items = .empty,
            .bodies = .empty,
            .statements = .empty,
            .expressions = .empty,
            .type_exprs = .empty,
            .patterns = .empty,
        };
    }

    pub fn reserve(self: *Builder) !void {
        const token_count = self.tree.tokens.len;
        const min_presized_tokens = 48;
        if (token_count < min_presized_tokens) return;

        const root_item_divisor: usize = if (token_count < 384) 128 else 384;
        const source_len = self.tree.source_text.len;
        const item_divisor: usize = if (source_len < 10_000) 24 else 32;
        try self.root_items.ensureTotalCapacityPrecise(self.allocator, @max(@as(usize, 1), token_count / root_item_divisor));
        try self.items.ensureTotalCapacityPrecise(self.allocator, @max(@as(usize, 1), token_count / item_divisor));
        try self.bodies.ensureTotalCapacityPrecise(self.allocator, @max(@as(usize, 1), token_count / 24));
        try self.statements.ensureTotalCapacityPrecise(self.allocator, @max(@as(usize, 1), token_count / 12));
        const expression_divisor: usize = if (token_count >= 256 and token_count < 384) 3 else 4;
        try self.expressions.ensureTotalCapacityPrecise(self.allocator, @max(@as(usize, 1), token_count / expression_divisor));
        const type_expr_divisor: usize = if ((token_count >= 256 and token_count < 384) or
            (token_count >= 1280 and token_count < 1600)) 10 else 8;
        try self.type_exprs.ensureTotalCapacityPrecise(self.allocator, @max(@as(usize, 1), token_count / type_expr_divisor));
        const pattern_divisor: usize = if ((token_count >= 90 and token_count < 112 and source_len >= 440 and source_len < 600) or
            (token_count >= 192 and token_count < 256 and (source_len >= 1000 and (source_len < 1500 or source_len >= 2048)))) 16 else 12;
        try self.patterns.ensureTotalCapacityPrecise(self.allocator, @max(@as(usize, 1), token_count / pattern_divisor));
    }

    pub fn finish(self: *Builder) void {
        self.file.root_items = self.root_items.items;
        self.file.items = self.items.items;
        self.file.bodies = self.bodies.items;
        self.file.statements = self.statements.items;
        self.file.expressions = self.expressions.items;
        self.file.type_exprs = self.type_exprs.items;
        self.file.patterns = self.patterns.items;
    }

    pub fn parseFile(self: *Builder) anyerror!void {
        try self.reserve();
        try self.lowerFileFromSyntax();
    }
};
