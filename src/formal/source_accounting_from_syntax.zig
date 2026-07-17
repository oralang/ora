//! Lossless source-formal site inventory from the parsed `SyntaxTree`.
//!
//! This traversal is intentionally independent from AST/sema verification-fact
//! collection. It uses syntax nodes, never source-text keyword matching.

const std = @import("std");
const accounting = @import("shared/source_accounting.zig");
const lexer = @import("ora_lexer");

pub const FormalTokenClass = union(enum) {
    source_site: accounting.SourceFactKind,
    expression_form,
    reserved_termination_measure,
    unrelated,
};

/// Pins the parser's formal-keyword surface independently from syntax-node
/// collection. `decreases`/`increases` remain reserved but are deliberately
/// not source-accounting sites until parser syntax exists for them.
pub fn classifyFormalToken(token: lexer.TokenType) FormalTokenClass {
    return switch (token) {
        .Requires => .{ .source_site = .requires },
        .Guard => .{ .source_site = .guard },
        .Ensures => .{ .source_site = .ensures },
        .EnsuresOk => .{ .source_site = .ensures_ok },
        .EnsuresErr => .{ .source_site = .ensures_err },
        .Invariant => .{ .source_site = .loop_invariant },
        .Assert => .{ .source_site = .assert },
        .Assume => .{ .source_site = .assume },
        .Havoc => .{ .source_site = .havoc },
        .Modifies => .{ .source_site = .modifies },
        .Old, .Result, .Forall, .Exists, .Ghost => .expression_form,
        .Decreases, .Increases => .reserved_termination_measure,
        else => .unrelated,
    };
}

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    declared_sites: []const accounting.DeclaredSite,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const Candidate = struct {
    owner: []const u8,
    range_start: u32,
    range_end: u32,
    kind: accounting.SourceFactKind,
    ordinal: u32 = 0,
    label: ?[]const u8 = null,
};

pub fn collect(
    allocator: std.mem.Allocator,
    root_node: anytype,
    canonical_path: []const u8,
) !Result {
    var result: Result = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .declared_sites = &.{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();
    var candidates: std.ArrayList(Candidate) = .empty;
    try collectNode(arena, &candidates, root_node, "module");

    std.mem.sort(Candidate, candidates.items, {}, lessCandidateForOrdinal);
    var previous_owner: ?[]const u8 = null;
    var previous_kind: ?accounting.SourceFactKind = null;
    var ordinal: u32 = 0;
    for (candidates.items) |*candidate| {
        if (previous_owner == null or previous_kind == null or
            !std.mem.eql(u8, previous_owner.?, candidate.owner) or previous_kind.? != candidate.kind)
        {
            ordinal = 0;
        }
        candidate.ordinal = ordinal;
        ordinal += 1;
        previous_owner = candidate.owner;
        previous_kind = candidate.kind;
    }
    std.mem.sort(Candidate, candidates.items, canonical_path, lessCandidateCanonical);

    const sites = try arena.alloc(accounting.DeclaredSite, candidates.items.len);
    for (candidates.items, sites, 0..) |candidate, *site, index| {
        site.* = .{
            .id = @intCast(index + 1),
            .key = .{
                .path = try arena.dupe(u8, canonical_path),
                .owner = candidate.owner,
                .range_start = candidate.range_start,
                .range_end = candidate.range_end,
                .kind = candidate.kind,
                .ordinal = candidate.ordinal,
            },
            .label = candidate.label,
        };
    }
    result.declared_sites = sites;
    return result;
}

fn collectNode(
    allocator: std.mem.Allocator,
    candidates: *std.ArrayList(Candidate),
    node: anytype,
    inherited_owner: []const u8,
) !void {
    var owner = inherited_owner;
    switch (node.kind()) {
        .ContractItem => owner = try namedOwner(allocator, inherited_owner, "contract", node, .Contract),
        .FunctionItem => owner = try namedOwner(allocator, inherited_owner, "function", node, .Fn),
        .TraitMethodSignature => owner = try namedOwner(allocator, inherited_owner, "trait_method", node, .Fn),
        .ImplItem => owner = try implOwner(allocator, inherited_owner, node),
        .WhileStmt => owner = try structuralOwner(allocator, inherited_owner, "while", node.range().start),
        .ForStmt => owner = try structuralOwner(allocator, inherited_owner, "for", node.range().start),
        .SwitchStmt => owner = try structuralOwner(allocator, inherited_owner, "switch", node.range().start),
        else => {},
    }

    if (factKindForNode(node)) |kind| {
        const range = node.range();
        try candidates.append(allocator, .{
            .owner = try allocator.dupe(u8, owner),
            .range_start = range.start,
            .range_end = range.end,
            .kind = kind,
            .label = if (kind == .loop_invariant or kind == .contract_invariant)
                try invariantLabel(allocator, node)
            else
                null,
        });
    }

    var children = node.children();
    while (children.next()) |child| switch (child) {
        .node => |child_node| try collectNode(allocator, candidates, child_node, owner),
        .token => {},
    };
}

fn implOwner(allocator: std.mem.Allocator, parent: []const u8, node: anytype) ![]const u8 {
    var names: [2][]const u8 = undefined;
    var name_count: usize = 0;
    var children = node.children();
    while (children.next()) |child| switch (child) {
        .node => {},
        .token => |token| {
            if (!isIdentifierLike(token.kind())) continue;
            if (name_count < names.len) names[name_count] = token.text();
            name_count += 1;
        },
    };
    if (name_count < names.len) return structuralOwner(allocator, parent, "impl", node.range().start);
    return std.fmt.allocPrint(allocator, "{s}/impl:{s}:{s}", .{ parent, names[0], names[1] });
}

fn isIdentifierLike(kind: lexer.TokenType) bool {
    if (lexer.isBuiltinTypeKeyword(kind)) return true;
    return switch (kind) {
        .Identifier, .Init, .From, .To, .Error, .Result, .Map, .Slice => true,
        else => false,
    };
}

fn factKindForNode(node: anytype) ?accounting.SourceFactKind {
    return switch (node.kind()) {
        .SpecClause => specClauseKind(node),
        .ContractInvariantItem => .contract_invariant,
        .InvariantClause => .loop_invariant,
        .AssertStmt => .assert,
        .AssumeStmt => .assume,
        .HavocStmt => .havoc,
        else => null,
    };
}

fn specClauseKind(node: anytype) ?accounting.SourceFactKind {
    var children = node.children();
    while (children.next()) |child| switch (child) {
        .node => {},
        .token => |token| return switch (token.kind()) {
            .Requires => .requires,
            .Guard => .guard,
            .Ensures => .ensures,
            .EnsuresOk => .ensures_ok,
            .EnsuresErr => .ensures_err,
            .Modifies => .modifies,
            else => null,
        },
    };
    return null;
}

fn namedOwner(
    allocator: std.mem.Allocator,
    parent: []const u8,
    kind: []const u8,
    node: anytype,
    marker: anytype,
) ![]const u8 {
    var saw_marker = false;
    var children = node.children();
    while (children.next()) |child| switch (child) {
        .node => {},
        .token => |token| {
            // Declaration names are contextual in Ora: a token such as
            // `slice` may retain its keyword token kind while the parser
            // accepts it as the function name. The first direct token after
            // `fn` is the name; requiring `.Identifier` silently split the
            // syntax and semantic owner identities.
            if (saw_marker) {
                return std.fmt.allocPrint(allocator, "{s}/{s}:{s}", .{ parent, kind, token.text() });
            }
            if (token.kind() == marker) saw_marker = true;
        },
    };
    return structuralOwner(allocator, parent, kind, node.range().start);
}

fn structuralOwner(allocator: std.mem.Allocator, parent: []const u8, kind: []const u8, start: u32) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}@{d}", .{ parent, kind, start });
}

fn invariantLabel(allocator: std.mem.Allocator, node: anytype) !?[]const u8 {
    var direct_expr_count: usize = 0;
    var first_name: ?@TypeOf(node) = null;
    var children = node.children();
    while (children.next()) |child| switch (child) {
        .token => {},
        .node => |child_node| if (isExpressionNode(child_node.kind())) {
            direct_expr_count += 1;
            if (direct_expr_count == 1 and child_node.kind() == .NameExpr) first_name = child_node;
        },
    };
    if (direct_expr_count < 2 or first_name == null) return null;
    const label: []const u8 = try allocator.dupe(u8, first_name.?.tree.sourceSlice(first_name.?.range()));
    return label;
}

fn isExpressionNode(kind: anytype) bool {
    return switch (kind) {
        .BinaryExpr,
        .RangeExpr,
        .UnaryExpr,
        .CallExpr,
        .FieldExpr,
        .IndexExpr,
        .GroupExpr,
        .TupleExpr,
        .ArrayLiteral,
        .StructLiteral,
        .SwitchExpr,
        .MatchExpr,
        .ExternalProxyExpr,
        .QuantifiedExpr,
        .OldExpr,
        .BuiltinExpr,
        .ErrorReturnExpr,
        .Literal,
        .NameExpr,
        .TryExpr,
        .ErrorExpr,
        .ComptimeExpr,
        => true,
        else => false,
    };
}

fn lessCandidateForOrdinal(_: void, lhs: Candidate, rhs: Candidate) bool {
    const owner_order = std.mem.order(u8, lhs.owner, rhs.owner);
    if (owner_order != .eq) return owner_order == .lt;
    if (lhs.kind != rhs.kind) return @intFromEnum(lhs.kind) < @intFromEnum(rhs.kind);
    if (lhs.range_start != rhs.range_start) return lhs.range_start < rhs.range_start;
    return lhs.range_end < rhs.range_end;
}

fn lessCandidateCanonical(path: []const u8, lhs: Candidate, rhs: Candidate) bool {
    _ = path;
    const owner_order = std.mem.order(u8, lhs.owner, rhs.owner);
    if (owner_order != .eq) return owner_order == .lt;
    if (lhs.range_start != rhs.range_start) return lhs.range_start < rhs.range_start;
    if (lhs.range_end != rhs.range_end) return lhs.range_end < rhs.range_end;
    if (lhs.kind != rhs.kind) return @intFromEnum(lhs.kind) < @intFromEnum(rhs.kind);
    return lhs.ordinal < rhs.ordinal;
}

test "formal keyword classification and lexer vocabulary are pinned" {
    // Adding any token forces review of the formal-keyword classifier instead
    // of allowing a new proof keyword to enter through `else` unnoticed.
    try std.testing.expectEqual(@as(usize, 146), std.meta.fields(lexer.TokenType).len);
    try std.testing.expectEqual(FormalTokenClass{ .source_site = .requires }, classifyFormalToken(.Requires));
    try std.testing.expectEqual(FormalTokenClass{ .source_site = .loop_invariant }, classifyFormalToken(.Invariant));
    try std.testing.expectEqual(FormalTokenClass.reserved_termination_measure, std.meta.activeTag(classifyFormalToken(.Decreases)));
    try std.testing.expectEqual(FormalTokenClass.expression_form, std.meta.activeTag(classifyFormalToken(.Old)));
}
