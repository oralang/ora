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
            .root_items = .{},
            .items = .{},
            .bodies = .{},
            .statements = .{},
            .expressions = .{},
            .type_exprs = .{},
            .patterns = .{},
        };
    }

    pub fn finish(self: *Builder) !void {
        self.file.root_items = try self.root_items.toOwnedSlice(self.allocator);
        self.file.items = try self.items.toOwnedSlice(self.allocator);
        self.file.bodies = try self.bodies.toOwnedSlice(self.allocator);
        self.file.statements = try self.statements.toOwnedSlice(self.allocator);
        self.file.expressions = try self.expressions.toOwnedSlice(self.allocator);
        self.file.type_exprs = try self.type_exprs.toOwnedSlice(self.allocator);
        self.file.patterns = try self.patterns.toOwnedSlice(self.allocator);
    }

    pub fn parseFile(self: *Builder) anyerror!void {
        try self.lowerFileFromSyntax();
    }
};
