const std = @import("std");
const lexer = @import("ora_lexer");
const diagnostics = @import("../diagnostics/mod.zig");
const source = @import("../source/mod.zig");
const green = @import("green.zig");
const SyntaxKind = @import("kinds.zig").SyntaxKind;

pub const ParseResult = struct {
    tree: green.SyntaxTree,
    diagnostics: diagnostics.DiagnosticList,

    pub fn deinit(self: *ParseResult) void {
        self.tree.deinit();
        self.diagnostics.deinit();
    }
};

pub fn parse(allocator: std.mem.Allocator, file_id: source.FileId, source_text: []const u8) !ParseResult {
    var result = ParseResult{
        .tree = undefined,
        .diagnostics = diagnostics.DiagnosticList.init(allocator),
    };
    errdefer result.diagnostics.deinit();

    var lexer_config = lexer.LexerConfig.development();
    lexer_config.minimum_diagnostic_severity = .Hint;

    var lex = try lexer.Lexer.initWithConfig(allocator, source_text, lexer_config);
    defer lex.deinit();

    const token_slice = try lex.scanTokens();
    defer allocator.free(token_slice);
    const trivia_slice = lex.getTrivia();

    for (lex.getDiagnostics()) |diag| {
        const diag_range = source.TextRange.init(diag.range.start_offset, diag.range.end_offset);
        const location = source.SourceLocation{ .file_id = file_id, .range = diag_range };
        try result.diagnostics.append(.Error, diag.message, location);
    }

    const copied_trivia = try copyTrivia(allocator, trivia_slice);
    errdefer allocator.free(copied_trivia);

    const copied_tokens = try copyTokens(allocator, token_slice);

    var parser = Parser.init(allocator, file_id, source_text, copied_trivia, copied_tokens, &result.diagnostics);
    result.tree = try parser.parseTree();
    return result;
}

const ChildRef = green.GreenChild;

const Parser = struct {
    allocator: std.mem.Allocator,
    file_id: source.FileId,
    source_text: []const u8,
    trivia: []green.GreenTrivia,
    tokens: std.ArrayList(green.GreenToken),
    diagnostics: *diagnostics.DiagnosticList,
    nodes: std.ArrayList(green.GreenNode),
    children: std.ArrayList(ChildRef),
    index: usize,
    pending_type_gt: u8,

    fn init(
        allocator: std.mem.Allocator,
        file_id: source.FileId,
        source_text: []const u8,
        trivia: []green.GreenTrivia,
        tokens: []green.GreenToken,
        diags: *diagnostics.DiagnosticList,
    ) Parser {
        return .{
            .allocator = allocator,
            .file_id = file_id,
            .source_text = source_text,
            .trivia = trivia,
            .tokens = std.ArrayList(green.GreenToken).fromOwnedSlice(tokens),
            .diagnostics = diags,
            .nodes = .{},
            .children = .{},
            .index = 0,
            .pending_type_gt = 0,
        };
    }

    fn parseTree(self: *Parser) anyerror!green.SyntaxTree {
        defer self.tokens.deinit(self.allocator);
        defer self.nodes.deinit(self.allocator);
        defer self.children.deinit(self.allocator);

        var root_children: std.ArrayList(ChildRef) = .{};
        defer root_children.deinit(self.allocator);

        while (!self.at(.Eof)) {
            const child = try self.parseTopLevelElement();
            try root_children.append(self.allocator, child);
        }

        const root = try self.finishNode(SyntaxKind.SourceFile, root_children.items);
        return .{
            .allocator = self.allocator,
            .file_id = self.file_id,
            .source_text = self.source_text,
            .trivia = self.trivia,
            .tokens = try self.tokens.toOwnedSlice(self.allocator),
            .children = try self.children.toOwnedSlice(self.allocator),
            .nodes = try self.nodes.toOwnedSlice(self.allocator),
            .root = root,
        };
    }

    fn parseTopLevelElement(self: *Parser) anyerror!ChildRef {
        if (self.startsTopLevelItem()) {
            return .{ .node = try self.parseTopLevelItem() };
        }
        return try self.parseElement(null);
    }

    fn parseElement(self: *Parser, closing: ?green.TokenKind) anyerror!ChildRef {
        if (closing) |expected| {
            if (self.at(expected)) {
                return .{ .node = try self.parseUnexpectedCloser() };
            }
        }

        return switch (self.current().kind) {
            .LeftParen => .{ .node = try self.parseDelimited(SyntaxKind.GroupParen, .RightParen) },
            .LeftBrace => .{ .node = try self.parseDelimited(SyntaxKind.GroupBrace, .RightBrace) },
            .LeftBracket => .{ .node = try self.parseDelimited(SyntaxKind.GroupBracket, .RightBracket) },
            .RightParen, .RightBrace, .RightBracket => .{ .node = try self.parseUnexpectedCloser() },
            else => .{ .token = self.bump() },
        };
    }

    fn parseTopLevelItem(self: *Parser) anyerror!green.GreenNodeId {
        if (self.at(.Comptime) and self.peekKind(1) == .Const) {
            return self.parseConstOrImportItem();
        }
        return switch (self.current().kind) {
            .Contract => self.parseContractItem(),
            .Pub, .Fn => self.parseFunctionItem(),
            .Struct => self.parseStructItem(),
            .Bitfield => self.parseBitfieldItem(),
            .Enum => self.parseEnumItem(),
            .Log => self.parseLogDeclItem(),
            .Error => self.parseErrorDeclItem(),
            .Const => self.parseConstantItem(false),
            .Storage, .Memory, .Tstore, .Let, .Var, .Immutable => self.parseFieldItem(),
            .Ghost => self.parseGhostItem(),
            else => self.parseErrorItemNode(true),
        };
    }

    fn parseContractMember(self: *Parser) anyerror!ChildRef {
        if (self.at(.RightBrace)) {
            return .{ .node = try self.parseUnexpectedCloser() };
        }

        if (self.at(.Invariant)) {
            return .{ .node = try self.parseContractInvariantItemNode() };
        }

        if (self.startsTopLevelItem()) {
            return .{ .node = try self.parseTopLevelItem() };
        }

        return .{ .node = try self.parseErrorItemNode(false) };
    }

    fn parseContractItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.LeftBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (!self.at(.LeftBrace)) {
            try self.reportHere("expected '{' after contract declaration");
            return self.finishNode(SyntaxKind.ContractItem, children.items);
        }

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            try children.append(self.allocator, try self.parseContractMember());
        }

        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated contract body", children.items);
        }

        return self.finishNode(SyntaxKind.ContractItem, children.items);
    }

    fn parseFunctionItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Pub)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }

        if (self.at(.Fn)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected 'fn' in function declaration");
        }

        while (!self.at(.Eof) and !self.at(.LeftParen) and !self.at(.LeftBrace) and !self.at(.Semicolon)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .node = try self.parseParameterListNode() });
        } else {
            try self.reportHere("expected parameter list in function declaration");
        }

        while (!self.at(.Eof) and !self.at(.LeftBrace) and !self.at(.Requires) and !self.at(.Ensures)) {
            if (self.at(.Semicolon)) break;
            if (self.at(.Arrow)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .LeftBrace, .Requires, .Ensures, .Semicolon }) });
                continue;
            }
            try children.append(self.allocator, try self.parseElement(null));
        }

        while (self.at(.Requires) or self.at(.Ensures)) {
            try children.append(self.allocator, .{ .node = try self.parseSpecClauseNode() });
        }

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (!self.at(.Eof)) {
            try self.reportHere("expected function body");
        }

        return self.finishNode(SyntaxKind.FunctionItem, children.items);
    }

    fn parseParameterListNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '(' to start parameter list");
            return self.finishNode(SyntaxKind.ParameterList, children.items);
        }

        while (!self.at(.Eof) and !self.at(.RightParen)) {
            try children.append(self.allocator, .{ .node = try self.parseParameterNode() });
            if (!self.at(.Comma)) break;
            try children.append(self.allocator, .{ .token = self.bump() });
        }

        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after parameter list");
        }

        return self.finishNode(SyntaxKind.ParameterList, children.items);
    }

    fn parseParameterNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        while (!self.at(.Eof) and !self.at(.Comma) and !self.at(.RightParen)) {
            if (self.at(.Colon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .RightParen }) });
                continue;
            }
            try children.append(self.allocator, try self.parseElement(null));
        }

        return self.finishNode(SyntaxKind.Parameter, children.items);
    }

    fn parseBodyNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '{' to start body");
            return self.finishNode(SyntaxKind.Body, children.items);
        }

        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseStatementNode() });
        }

        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated body", children.items);
        }

        return self.finishNode(SyntaxKind.Body, children.items);
    }

    fn parseSpecClauseNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Requires) or self.at(.Ensures)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected specification clause");
            return self.finishNode(SyntaxKind.SpecClause, children.items);
        }

        if (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.LeftBrace) and !self.at(.Requires) and !self.at(.Ensures)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .LeftBrace, .Requires, .Ensures }) });
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }

        return self.finishNode(SyntaxKind.SpecClause, children.items);
    }

    fn parseContractInvariantItemNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Invariant)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected contract invariant");
            return self.finishNode(SyntaxKind.ContractInvariantItem, children.items);
        }

        if (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .RightBrace }) });
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (self.at(.RightBrace)) {
            try self.reportHere("expected ';' before '}'");
        } else {
            try self.reportUnterminated("unterminated contract invariant", children.items);
        }

        return self.finishNode(SyntaxKind.ContractInvariantItem, children.items);
    }

    fn parseGhostItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else if (self.startsTopLevelItem()) {
            try children.append(self.allocator, .{ .node = try self.parseTopLevelItem() });
        } else {
            try children.append(self.allocator, .{ .node = try self.parseErrorItemNode(false) });
        }

        return self.finishNode(SyntaxKind.GhostItem, children.items);
    }

    fn parseConstOrImportItem(self: *Parser) anyerror!green.GreenNodeId {
        return if (self.looksLikeImportItem()) self.parseImportItem() else self.parseConstantItem(true);
    }

    fn parseImportItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Comptime)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }
        if (self.at(.Const)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            if (self.at(.Identifier)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected import alias");
            }
            if (self.at(.Equal)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected '=' after import alias");
            }
        }

        if (self.at(.At)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }
        if (self.at(.Import)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected import keyword");
        }
        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '(' after import");
        }
        if (self.at(.StringLiteral)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected import path string");
        }
        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after import path");
        }
        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' after import");
        }
        return self.finishNode(SyntaxKind.ImportItem, children.items);
    }

    fn parseConstantItem(self: *Parser, allow_comptime_prefix: bool) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (allow_comptime_prefix and self.at(.Comptime)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }
        if (self.at(.Const)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected 'const'");
            return self.finishNode(SyntaxKind.ConstantItem, children.items);
        }
        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected constant name");
        }
        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Equal, .Semicolon }) });
        }
        if (self.at(.Equal)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{.Semicolon}) });
        } else {
            try self.reportHere("expected '=' after constant name");
        }
        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' after constant");
        }
        return self.finishNode(SyntaxKind.ConstantItem, children.items);
    }

    fn parseFieldItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        while (!self.at(.Eof) and !self.at(.Semicolon)) {
            if (self.at(.Colon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Equal, .Semicolon }) });
                continue;
            }
            if (self.at(.Equal)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{.Semicolon}) });
                break;
            }
            try children.append(self.allocator, try self.parseElement(null));
        }
        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated field item", children.items);
        }
        return self.finishNode(SyntaxKind.FieldItem, children.items);
    }

    fn parseLogDeclItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected log name");
        }

        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '(' after log name");
            return self.finishNode(SyntaxKind.LogDeclItem, children.items);
        }

        while (!self.at(.Eof) and !self.at(.RightParen)) {
            try children.append(self.allocator, .{ .node = try self.parseLogFieldNode() });
            if (!self.at(.Comma)) break;
            try children.append(self.allocator, .{ .token = self.bump() });
        }

        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after log fields");
        }
        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' after log declaration");
        }
        return self.finishNode(SyntaxKind.LogDeclItem, children.items);
    }

    fn parseLogFieldNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (tokenIsIdentifierLike(self.current().kind) and std.mem.eql(u8, self.source_text[self.current().range.start..self.current().range.end], "indexed")) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }
        if (tokenIsIdentifierLike(self.current().kind)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected log field name");
            return self.finishNode(SyntaxKind.LogField, children.items);
        }
        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .RightParen }) });
        } else {
            try self.reportHere("expected ':' after log field name");
        }
        return self.finishNode(SyntaxKind.LogField, children.items);
    }

    fn parseErrorDeclItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected error name");
        }
        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .node = try self.parseParameterListNode() });
        }
        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' after error declaration");
        }
        return self.finishNode(SyntaxKind.ErrorDeclItem, children.items);
    }

    fn parseBracedItem(self: *Parser, kind: SyntaxKind) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.LeftBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }
        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseDelimited(SyntaxKind.GroupBrace, .RightBrace) });
        } else {
            try self.reportHere("expected braced body");
        }
        return self.finishNode(kind, children.items);
    }

    fn parseStructItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.LeftBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (!self.at(.LeftBrace)) {
            try self.reportHere("expected braced body");
            return self.finishNode(SyntaxKind.StructItem, children.items);
        }

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            if (self.at(.Comma) or self.at(.Semicolon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                continue;
            }
            try children.append(self.allocator, .{ .node = try self.parseStructFieldNode() });
        }

        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated braced item body", children.items);
        }

        return self.finishNode(SyntaxKind.StructItem, children.items);
    }

    fn parseBitfieldItem(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.LeftBrace)) {
            if (self.at(.Colon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{.LeftBrace}) });
                continue;
            }
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (!self.at(.LeftBrace)) {
            try self.reportHere("expected braced body");
            return self.finishNode(SyntaxKind.BitfieldItem, children.items);
        }

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            if (self.at(.Comma) or self.at(.Semicolon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                continue;
            }
            try children.append(self.allocator, .{ .node = try self.parseBitfieldFieldNode() });
        }

        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated braced item body", children.items);
        }

        return self.finishNode(SyntaxKind.BitfieldItem, children.items);
    }

    fn parseEnumItem(self: *Parser) anyerror!green.GreenNodeId {
        return self.parseMemberItem(SyntaxKind.EnumItem, SyntaxKind.EnumVariant, "expected enum variant");
    }

    fn parseMemberItem(self: *Parser, item_kind: SyntaxKind, member_kind: SyntaxKind, message: []const u8) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.LeftBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (!self.at(.LeftBrace)) {
            try self.reportHere("expected braced body");
            return self.finishNode(item_kind, children.items);
        }

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            if (self.at(.Comma) or self.at(.Semicolon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                continue;
            }
            try children.append(self.allocator, .{ .node = try self.parseMemberNode(member_kind, message) });
        }

        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated braced item body", children.items);
        }

        return self.finishNode(item_kind, children.items);
    }

    fn parseMemberNode(self: *Parser, kind: SyntaxKind, message: []const u8) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        while (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.Comma) and !self.at(.RightBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (children.items.len == 0) {
            try self.reportHere(message);
            return self.finishNode(kind, children.items);
        }

        return self.finishNode(kind, children.items);
    }

    fn parseStructFieldNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected struct field");
            return self.finishNode(SyntaxKind.StructField, children.items);
        }

        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .Semicolon, .RightBrace }) });
        } else {
            try self.reportHere("expected ':' after field name");
        }

        return self.finishNode(SyntaxKind.StructField, children.items);
    }

    fn parseBitfieldFieldNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected bitfield field");
            return self.finishNode(SyntaxKind.BitfieldField, children.items);
        }

        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .LeftParen, .At, .Semicolon, .RightBrace }) });
        } else {
            try self.reportHere("expected ':' after field name");
        }

        while (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.Comma) and !self.at(.RightBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        return self.finishNode(SyntaxKind.BitfieldField, children.items);
    }

    fn parseSemicolonOrBracedItem(self: *Parser, kind: SyntaxKind) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.LeftBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }
        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseDelimited(SyntaxKind.GroupBrace, .RightBrace) });
        } else if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' or braced body");
        }
        return self.finishNode(kind, children.items);
    }

    fn parseDelimitedItem(self: *Parser, kind: SyntaxKind, terminator: green.TokenKind) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        while (!self.at(.Eof) and !self.at(terminator)) {
            try children.append(self.allocator, try self.parseElement(null));
        }
        if (self.at(terminator)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated item", children.items);
        }
        return self.finishNode(kind, children.items);
    }

    fn parseDelimitedLike(
        self: *Parser,
        kind: SyntaxKind,
        opener: green.TokenKind,
        closing: green.TokenKind,
    ) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(opener)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected delimiter");
            return self.finishNode(kind, children.items);
        }

        while (!self.at(.Eof) and !self.at(closing)) {
            try children.append(self.allocator, try self.parseElement(closing));
        }

        if (self.at(closing)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated delimited node", children.items);
        }

        return self.finishNode(kind, children.items);
    }

    fn parseStatementNode(self: *Parser) anyerror!green.GreenNodeId {
        if (self.looksLikeLabeledBlockStmt()) return self.parseLabeledBlockStmtNode();
        if (self.looksLikeDestructuringAssignStmt()) return self.parseDestructuringAssignStmtNode();
        if (self.looksLikeDirectiveStmt("lock")) return self.parseDirectiveStmtNode(SyntaxKind.LockStmt, "lock");
        if (self.looksLikeDirectiveStmt("unlock")) return self.parseDirectiveStmtNode(SyntaxKind.UnlockStmt, "unlock");
        return switch (self.current().kind) {
            .LeftBrace => self.parseBlockStmtNode(),
            .If => self.parseIfStmtNode(),
            .While => self.parseWhileStmtNode(),
            .For => self.parseForStmtNode(),
            .Switch => self.parseSwitchStmtNode(),
            .Try => self.parseTryStmtNode(),
            .Return => self.parseReturnStmtNode(),
            .Log => self.parseLogStmtNode(),
            .Assert => self.parseAssertStmtNode(),
            .Assume => self.parseAssumeStmtNode(),
            .Havoc => self.parseDelimitedStatementNode(SyntaxKind.HavocStmt),
            .Break => self.parseDelimitedStatementNode(SyntaxKind.BreakStmt),
            .Continue => self.parseDelimitedStatementNode(SyntaxKind.ContinueStmt),
            .Storage, .Memory, .Tstore, .Let, .Var, .Immutable => self.parseVariableDeclStmtNode(),
            else => self.parseExprOrAssignStmtNode(),
        };
    }

    fn parseLabeledBlockStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected label name");
            return self.finishNode(SyntaxKind.LabeledBlockStmt, children.items);
        }

        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ':' after label");
            return self.finishNode(SyntaxKind.LabeledBlockStmt, children.items);
        }

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try self.reportHere("expected block after label");
        }

        return self.finishNode(SyntaxKind.LabeledBlockStmt, children.items);
    }

    fn parseBlockStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        return self.finishNode(SyntaxKind.BlockStmt, children.items);
    }

    fn parseIfStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftParen)) {
            try self.appendConditionExpr(&children);
        } else {
            try self.reportHere("expected '(' after if");
        }

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try self.reportHere("expected body after if condition");
        }

        if (self.at(.Else)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            if (self.at(.If)) {
                try children.append(self.allocator, .{ .node = try self.parseIfStmtNode() });
            } else if (self.at(.LeftBrace)) {
                try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
            } else {
                try self.reportHere("expected 'if' or body after else");
            }
        }

        return self.finishNode(SyntaxKind.IfStmt, children.items);
    }

    fn parseWhileStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftParen)) {
            try self.appendConditionExpr(&children);
        } else {
            try self.reportHere("expected '(' after while");
        }

        while (self.at(.Invariant)) {
            try children.append(self.allocator, .{ .node = try self.parseInvariantClauseNode() });
        }

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try self.reportHere("expected body after while condition");
        }

        return self.finishNode(SyntaxKind.WhileStmt, children.items);
    }

    fn parseForStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftParen)) {
            try self.appendConditionExpr(&children);
        } else {
            try self.reportHere("expected '(' after for");
        }

        while (!self.at(.Eof) and !self.at(.LeftBrace)) {
            if (self.at(.Semicolon) or self.at(.RightBrace) or self.at(.Invariant)) break;
            try children.append(self.allocator, try self.parseElement(null));
        }

        while (self.at(.Invariant)) {
            try children.append(self.allocator, .{ .node = try self.parseInvariantClauseNode() });
        }

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try self.reportHere("expected body after for clause");
        }

        return self.finishNode(SyntaxKind.ForStmt, children.items);
    }

    fn parseSwitchStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftParen)) {
            try self.appendConditionExpr(&children);
        } else {
            try self.reportHere("expected '(' after switch");
        }

        if (!self.at(.LeftBrace)) {
            try self.reportHere("expected '{' after switch condition");
            return self.finishNode(SyntaxKind.SwitchStmt, children.items);
        }

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseSwitchArmNode() });
            if (self.at(.Comma)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            }
        }
        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated switch body", children.items);
        }

        return self.finishNode(SyntaxKind.SwitchStmt, children.items);
    }

    fn parseTryStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try self.reportHere("expected body after try");
        }

        if (self.at(.Catch)) {
            try children.append(self.allocator, .{ .node = try self.parseCatchClauseNode() });
        }

        return self.finishNode(SyntaxKind.TryStmt, children.items);
    }

    fn parseCatchClauseNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Catch)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected catch");
            return self.finishNode(SyntaxKind.CatchClause, children.items);
        }

        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .node = try self.parseDelimited(SyntaxKind.GroupParen, .RightParen) });
        }

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try self.reportHere("expected catch body");
        }

        return self.finishNode(SyntaxKind.CatchClause, children.items);
    }

    fn parseInvariantClauseNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Invariant)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected invariant clause");
            return self.finishNode(SyntaxKind.InvariantClause, children.items);
        }

        if (!self.at(.Semicolon) and !self.at(.LeftBrace) and !self.at(.Invariant) and !self.at(.Eof)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .LeftBrace, .Invariant }) });
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }

        return self.finishNode(SyntaxKind.InvariantClause, children.items);
    }

    fn parseSwitchArmNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = try self.parseSwitchPatternExprNode() });

        if (self.at(.Arrow)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '=>' in switch arm");
            return self.finishNode(SyntaxKind.SwitchArm, children.items);
        }

        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try children.append(self.allocator, .{ .node = try self.parseExprStmtNode() });
        }

        return self.finishNode(SyntaxKind.SwitchArm, children.items);
    }

    fn parseReturnStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (!self.at(.Semicolon) and !self.at(.RightBrace) and !self.at(.Eof)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .RightBrace }) });
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (self.at(.RightBrace)) {
            try self.reportHere("expected ';' before '}'");
        } else {
            try self.reportUnterminated("unterminated return statement", children.items);
        }

        return self.finishNode(SyntaxKind.ReturnStmt, children.items);
    }

    fn parseLogStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected log name");
        }

        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            while (!self.at(.Eof) and !self.at(.RightParen)) {
                try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen }) });
                if (!self.at(.Comma)) break;
                try children.append(self.allocator, .{ .token = self.bump() });
            }
            if (self.at(.RightParen)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected ')' after log arguments");
            }
        } else {
            try self.reportHere("expected '(' after log name");
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' after log statement");
        }

        return self.finishNode(SyntaxKind.LogStmt, children.items);
    }

    fn parseAssertStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        return self.parseParenExprStatementNode(SyntaxKind.AssertStmt, "assert");
    }

    fn parseAssumeStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        return self.parseParenExprStatementNode(SyntaxKind.AssumeStmt, "assume");
    }

    fn parseParenExprStatementNode(self: *Parser, kind: SyntaxKind, keyword: []const u8) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (!self.at(.LeftParen)) {
            _ = keyword;
            try self.reportHere("expected '(' after statement keyword");
            return self.finishNode(kind, children.items);
        }

        try children.append(self.allocator, .{ .token = self.bump() });
        if (!self.at(.RightParen)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen }) });
            while (self.at(.Comma)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                if (self.at(.RightParen)) break;
                try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen }) });
            }
        }

        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after statement arguments");
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' after statement");
        }

        return self.finishNode(kind, children.items);
    }

    fn parseVariableDeclStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        while (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.Equal) and !self.at(.RightBrace)) {
            if (self.at(.Colon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Equal, .Semicolon, .RightBrace }) });
                continue;
            }
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (self.at(.Equal)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .RightBrace }) });
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (self.at(.RightBrace)) {
            try self.reportHere("expected ';' before '}'");
        } else {
            try self.reportUnterminated("unterminated variable declaration", children.items);
        }

        return self.finishNode(SyntaxKind.VariableDeclStmt, children.items);
    }

    fn parseDestructuringAssignStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        while (!self.at(.Eof) and !(self.at(.Dot) and self.peekKind(1) == .LeftBrace) and !self.at(.Semicolon) and !self.at(.RightBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (self.at(.Dot) and self.peekKind(1) == .LeftBrace) {
            try children.append(self.allocator, .{ .node = try self.parseDestructuringPatternNode() });
        } else {
            try self.reportHere("expected destructuring pattern");
            return self.finishNode(SyntaxKind.DestructuringAssignStmt, children.items);
        }

        if (self.at(.Equal)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .RightBrace }) });
        } else {
            try self.reportHere("expected '=' after destructuring pattern");
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (self.at(.RightBrace)) {
            try self.reportHere("expected ';' before '}'");
        } else {
            try self.reportUnterminated("unterminated destructuring assignment", children.items);
        }

        return self.finishNode(SyntaxKind.DestructuringAssignStmt, children.items);
    }

    fn parseDestructuringPatternNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '{' after '.' in destructuring pattern");
            return self.finishNode(SyntaxKind.DestructuringPattern, children.items);
        }

        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseDestructuringFieldNode() });
            if (self.at(.Comma)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else break;
        }

        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated destructuring pattern", children.items);
        }

        return self.finishNode(SyntaxKind.DestructuringPattern, children.items);
    }

    fn parseDestructuringFieldNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected field name in destructuring pattern");
            return self.finishNode(SyntaxKind.DestructuringField, children.items);
        }

        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            if (self.at(.Identifier)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected binding name after ':' in destructuring pattern");
            }
        }

        return self.finishNode(SyntaxKind.DestructuringField, children.items);
    }

    fn parseDelimitedStatementNode(self: *Parser, kind: SyntaxKind) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        while (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.RightBrace)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (self.at(.RightBrace)) {
            try self.reportHere("expected ';' before '}'");
        } else {
            try self.reportUnterminated("unterminated statement", children.items);
        }

        return self.finishNode(kind, children.items);
    }

    fn parseDirectiveStmtNode(self: *Parser, kind: SyntaxKind, comptime name: []const u8) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.At)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '@' before directive");
            return self.finishNode(kind, children.items);
        }

        if (self.at(.Identifier) and std.mem.eql(u8, self.source_text[self.current().range.start..self.current().range.end], name)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected directive name");
            return self.finishNode(kind, children.items);
        }

        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            if (!self.at(.RightParen)) {
                try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{.RightParen}) });
            }
            if (self.at(.RightParen)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected ')' after directive argument");
            }
        } else {
            try self.reportHere("expected '(' after directive");
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (self.at(.RightBrace)) {
            try self.reportHere("expected ';' before '}'");
        } else {
            try self.reportUnterminated("unterminated directive statement", children.items);
        }

        return self.finishNode(kind, children.items);
    }

    fn parseExprOrAssignStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .RightBrace }) });

        if (isAssignmentToken(self.current().kind)) {
            const assign_kind = self.current().kind;
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .RightBrace }) });
            if (self.at(.Semicolon)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else if (self.at(.RightBrace)) {
                try self.reportHere("expected ';' before '}'");
            } else {
                try self.reportUnterminated("unterminated assignment statement", children.items);
            }
            return self.finishNode(if (assign_kind == .Equal) SyntaxKind.AssignStmt else SyntaxKind.CompoundAssignStmt, children.items);
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else if (self.at(.RightBrace)) {
            try self.reportHere("expected ';' before '}'");
        } else {
            try self.reportUnterminated("unterminated expression statement", children.items);
        }

        return self.finishNode(SyntaxKind.ExprStmt, children.items);
    }

    fn parseExprStmtNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Semicolon, .Comma, .RightBrace }) });
        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }
        return self.finishNode(SyntaxKind.ExprStmt, children.items);
    }

    fn parseExpressionNode(self: *Parser, terminators: []const green.TokenKind) anyerror!green.GreenNodeId {
        return self.parseBinaryExprNode(0, terminators);
    }

    fn parseBinaryExprNode(self: *Parser, min_precedence: u8, terminators: []const green.TokenKind) anyerror!green.GreenNodeId {
        var lhs = try self.parseUnaryExprNode(terminators);
        while (binaryOpPrecedence(self.current().kind)) |precedence| {
            if (precedence < min_precedence or self.atAny(terminators)) break;

            const op_token = self.bump();
            const rhs = try self.parseBinaryExprNode(precedence + 1, terminators);

            const expr_children = [_]ChildRef{
                .{ .node = lhs },
                .{ .token = op_token },
                .{ .node = rhs },
            };
            lhs = try self.finishNode(SyntaxKind.BinaryExpr, &expr_children);
        }
        return lhs;
    }

    fn parseUnaryExprNode(self: *Parser, terminators: []const green.TokenKind) anyerror!green.GreenNodeId {
        if (self.atAny(terminators)) {
            return self.parseExpressionErrorNode("expected expression");
        }

        return switch (self.current().kind) {
            .Bang, .Minus, .Plus, .Try => blk: {
                const op_token = self.bump();
                const operand = try self.parseUnaryExprNode(terminators);
                const expr_children = [_]ChildRef{
                    .{ .token = op_token },
                    .{ .node = operand },
                };
                break :blk self.finishNode(SyntaxKind.UnaryExpr, &expr_children);
            },
            else => self.parsePostfixExprNode(terminators),
        };
    }

    fn parsePostfixExprNode(self: *Parser, terminators: []const green.TokenKind) anyerror!green.GreenNodeId {
        var expr = try self.parsePrimaryExprNode(terminators);

        while (!self.atAny(terminators)) {
            if (self.at(.LeftParen)) {
                expr = try self.parseCallExprNode(expr, terminators);
                continue;
            }
            if (self.at(.Dot)) {
                expr = try self.parseFieldExprNode(expr);
                continue;
            }
            if (self.at(.LeftBracket)) {
                expr = try self.parseIndexExprNode(expr);
                continue;
            }
            if (self.at(.LeftBrace) and self.nodeCouldStartStructLiteral(expr)) {
                expr = try self.parseStructLiteralExprNode(expr, terminators);
                continue;
            }
            break;
        }

        return expr;
    }

    fn parsePrimaryExprNode(self: *Parser, terminators: []const green.TokenKind) anyerror!green.GreenNodeId {
        _ = terminators;
        return switch (self.current().kind) {
            .Identifier, .Result, .From, .To, .Error => self.parseSingleTokenExprNode(SyntaxKind.NameExpr),
            .IntegerLiteral, .BinaryLiteral, .HexLiteral, .AddressLiteral, .BytesLiteral, .StringLiteral, .RawStringLiteral, .CharacterLiteral, .True, .False => self.parseSingleTokenExprNode(SyntaxKind.Literal),
            .LeftParen => self.parseParenLikeExprNode(),
            .LeftBracket => self.parseArrayLiteralExprNode(),
            .Comptime => self.parseComptimeExprNode(),
            .Old => self.parseOldExprNode(),
            .Forall, .Exists => self.parseQuantifiedExprNode(),
            .At => self.parseBuiltinExprNode(),
            .Switch => self.parseSwitchExprNode(),
            else => self.parseExpressionErrorNode("expected expression"),
        };
    }

    fn parseComptimeExprNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseBodyNode() });
        } else {
            try self.reportHere("expected body after comptime");
        }
        return self.finishNode(SyntaxKind.ComptimeExpr, children.items);
    }

    fn parseTypeExprNode(self: *Parser, stops: []const green.TokenKind) anyerror!green.GreenNodeId {
        if (self.atAny(stops)) return self.parseTypeErrorNode("expected type expression");
        return self.parseTypeExprInnerNode(stops);
    }

    fn parseTypeExprInnerNode(self: *Parser, stops: []const green.TokenKind) anyerror!green.GreenNodeId {
        if (self.at(.Bang)) {
            var children: std.ArrayList(ChildRef) = .{};
            defer children.deinit(self.allocator);

            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypePrimaryNode(stops) });
            while (self.at(.Pipe)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                try children.append(self.allocator, .{ .node = try self.parseTypePrimaryNode(stops) });
            }
            return self.finishNode(SyntaxKind.ErrorUnionType, children.items);
        }
        return self.parseTypePrimaryNode(stops);
    }

    fn parseTypePrimaryNode(self: *Parser, stops: []const green.TokenKind) anyerror!green.GreenNodeId {
        _ = stops;
        return switch (self.current().kind) {
            .LeftParen => self.parseTupleTypeNode(),
            .LeftBracket => self.parseArrayTypeNode(),
            .Slice => self.parseSliceTypeNode(),
            .Struct => self.parseAnonymousStructTypeNode(),
            .Map,
            .Identifier,
            .Error,
            .Result,
            .From,
            .To,
            .U8,
            .U16,
            .U32,
            .U64,
            .U128,
            .U256,
            .I8,
            .I16,
            .I32,
            .I64,
            .I128,
            .I256,
            .Bool,
            .Address,
            .String,
            .Bytes,
            .Void,
            => self.parsePathOrGenericTypeNode(),
            else => self.parseTypeErrorNode("expected type expression"),
        };
    }

    fn parseTupleTypeNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (!self.at(.RightParen)) {
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .RightParen }) });
            while (self.at(.Comma)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                if (self.at(.RightParen)) break;
                try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .RightParen }) });
            }
        }
        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after tuple type");
        }
        return self.finishNode(SyntaxKind.TupleType, children.items);
    }

    fn parseArrayTypeNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Semicolon, .RightBracket }) });
        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ';' in array type");
        }
        if (self.at(.IntegerLiteral) or self.at(.BinaryLiteral) or self.at(.HexLiteral)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected array size");
        }
        if (self.at(.RightBracket)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ']' after array type");
        }
        return self.finishNode(SyntaxKind.ArrayType, children.items);
    }

    fn parseSliceTypeNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftBracket)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{.RightBracket}) });
            if (self.at(.RightBracket)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected ']' after slice element type");
            }
        } else {
            try self.reportHere("expected '[' after slice");
        }
        return self.finishNode(SyntaxKind.SliceType, children.items);
    }

    fn parseAnonymousStructTypeNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            while (!self.at(.Eof) and !self.at(.RightBrace)) {
                if (self.at(.Comma) or self.at(.Semicolon)) {
                    try children.append(self.allocator, .{ .token = self.bump() });
                    continue;
                }
                try children.append(self.allocator, .{ .node = try self.parseAnonymousStructFieldNode() });
            }
            if (self.at(.RightBrace)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected '}' after anonymous struct type");
            }
        } else {
            try self.reportHere("expected '{' after struct in type");
        }
        return self.finishNode(SyntaxKind.AnonymousStructType, children.items);
    }

    fn parseAnonymousStructFieldNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected field name in anonymous struct type");
            return self.finishNode(SyntaxKind.AnonymousStructField, children.items);
        }
        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .Semicolon, .RightBrace }) });
        } else {
            try self.reportHere("expected ':' after field name");
        }
        return self.finishNode(SyntaxKind.AnonymousStructField, children.items);
    }

    fn parsePathOrGenericTypeNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.Less)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            while (!self.at(.Eof) and !self.typeAtGreaterToken()) {
                if (self.at(.IntegerLiteral) or self.at(.BinaryLiteral) or self.at(.HexLiteral)) {
                    try children.append(self.allocator, .{ .token = self.bump() });
                } else {
                    try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .GreaterGreater }) });
                }
                if (!self.at(.Comma)) break;
                try children.append(self.allocator, .{ .token = self.bump() });
            }
            if (self.typeAtGreaterToken()) {
                try children.append(self.allocator, .{ .token = self.bumpTypeGreater() });
            } else {
                try self.reportHere("expected '>' after generic arguments");
            }
            return self.finishNode(SyntaxKind.GenericType, children.items);
        }
        return self.finishNode(SyntaxKind.PathType, children.items);
    }

    fn parseTypeErrorNode(self: *Parser, message: []const u8) anyerror!green.GreenNodeId {
        try self.reportHere(message);
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);
        if (!self.at(.Eof)) {
            try children.append(self.allocator, try self.parseElement(null));
        }
        return self.finishNode(SyntaxKind.Error, children.items);
    }

    fn parseSingleTokenExprNode(self: *Parser, kind: SyntaxKind) anyerror!green.GreenNodeId {
        const expr_children = [_]ChildRef{.{ .token = self.bump() }};
        return self.finishNode(kind, &expr_children);
    }

    fn parseParenLikeExprNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            return self.finishNode(SyntaxKind.TupleExpr, children.items);
        }

        try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen }) });
        if (self.at(.Comma)) {
            while (self.at(.Comma)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                if (self.at(.RightParen)) break;
                try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen }) });
            }
            if (self.at(.RightParen)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected ')' after tuple expression");
            }
            return self.finishNode(SyntaxKind.TupleExpr, children.items);
        }

        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after grouped expression");
        }
        return self.finishNode(SyntaxKind.GroupExpr, children.items);
    }

    fn parseArrayLiteralExprNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBracket)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightBracket }) });
            if (!self.at(.Comma)) break;
            try children.append(self.allocator, .{ .token = self.bump() });
        }

        if (self.at(.RightBracket)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ']' after array literal");
        }

        return self.finishNode(SyntaxKind.ArrayLiteral, children.items);
    }

    fn parseOldExprNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{.RightParen}) });
            if (self.at(.RightParen)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected ')' after old expression");
            }
        } else {
            try self.reportHere("expected '(' after old");
        }

        return self.finishNode(SyntaxKind.OldExpr, children.items);
    }

    fn parseBuiltinExprNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        var builtin_name: ?[]const u8 = null;
        if (self.at(.Identifier)) {
            const name_token = self.bump();
            builtin_name = self.source_text[self.tokens.items[name_token.index()].range.start..self.tokens.items[name_token.index()].range.end];
            try children.append(self.allocator, .{ .token = name_token });
        } else {
            try self.reportHere("expected builtin name after '@'");
        }

        if (self.at(.LeftParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            if (builtin_name != null and std.mem.eql(u8, builtin_name.?, "cast") and !self.at(.RightParen)) {
                try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Comma, .RightParen }) });
                if (self.at(.Comma)) {
                    try children.append(self.allocator, .{ .token = self.bump() });
                } else if (!self.at(.RightParen)) {
                    try self.reportHere("expected ',' after builtin cast type");
                }
            }
            while (!self.at(.Eof) and !self.at(.RightParen)) {
                try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen }) });
                if (!self.at(.Comma)) break;
                try children.append(self.allocator, .{ .token = self.bump() });
            }
            if (self.at(.RightParen)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            } else {
                try self.reportHere("expected ')' after builtin arguments");
            }
        } else {
            try self.reportHere("expected '(' after builtin name");
        }

        return self.finishNode(SyntaxKind.BuiltinExpr, children.items);
    }

    fn parseQuantifiedExprNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected quantified variable name");
            return self.finishNode(SyntaxKind.QuantifiedExpr, children.items);
        }

        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseTypeExprNode(&.{ .Where, .Arrow }) });
        } else {
            try self.reportHere("expected ':' after quantified variable");
            return self.finishNode(SyntaxKind.QuantifiedExpr, children.items);
        }

        if (self.at(.Where)) {
            try children.append(self.allocator, .{ .token = self.bump() });
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{.Arrow}) });
        }

        if (self.at(.Arrow)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '=>' after quantified expression");
            return self.finishNode(SyntaxKind.QuantifiedExpr, children.items);
        }

        try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen, .RightBracket, .RightBrace, .Semicolon }) });
        return self.finishNode(SyntaxKind.QuantifiedExpr, children.items);
    }

    fn parseSwitchExprNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.LeftParen)) {
            try self.appendConditionExpr(&children);
        } else {
            try self.reportHere("expected '(' after switch");
        }

        if (!self.at(.LeftBrace)) {
            try self.reportHere("expected '{' after switch condition");
            return self.finishNode(SyntaxKind.SwitchExpr, children.items);
        }

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .node = try self.parseSwitchExprArmNode() });
            if (self.at(.Comma)) {
                try children.append(self.allocator, .{ .token = self.bump() });
            }
        }
        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated switch expression", children.items);
        }
        return self.finishNode(SyntaxKind.SwitchExpr, children.items);
    }

    fn parseSwitchExprArmNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = try self.parseSwitchPatternExprNode() });
        if (self.at(.Arrow)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '=>' in switch expression arm");
            return self.finishNode(SyntaxKind.SwitchExprArm, children.items);
        }

        try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightBrace }) });
        return self.finishNode(SyntaxKind.SwitchExprArm, children.items);
    }

    fn parseSwitchPatternExprNode(self: *Parser) anyerror!green.GreenNodeId {
        const start_expr = try self.parseExpressionNode(&.{ .Arrow, .DotDot, .DotDotDot, .RightBrace });
        if (self.at(.DotDot) or self.at(.DotDotDot)) {
            const children = [_]ChildRef{
                .{ .node = start_expr },
                .{ .token = self.bump() },
                .{ .node = try self.parseExpressionNode(&.{ .Arrow, .RightBrace }) },
            };
            return self.finishNode(SyntaxKind.RangeExpr, &children);
        }
        return start_expr;
    }

    fn parseCallExprNode(self: *Parser, callee: green.GreenNodeId, terminators: []const green.TokenKind) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = callee });
        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightParen)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightParen }) });
            if (!self.at(.Comma)) break;
            try children.append(self.allocator, .{ .token = self.bump() });
        }
        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after call arguments");
        }
        _ = terminators;
        return self.finishNode(SyntaxKind.CallExpr, children.items);
    }

    fn parseFieldExprNode(self: *Parser, base: green.GreenNodeId) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = base });
        try children.append(self.allocator, .{ .token = self.bump() });
        if (self.at(.Identifier) or self.at(.Error) or self.at(.From) or self.at(.To)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected field name after '.'");
        }
        return self.finishNode(SyntaxKind.FieldExpr, children.items);
    }

    fn parseIndexExprNode(self: *Parser, base: green.GreenNodeId) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = base });
        try children.append(self.allocator, .{ .token = self.bump() });
        try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{.RightBracket}) });
        if (self.at(.RightBracket)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ']' after index expression");
        }

        return self.finishNode(SyntaxKind.IndexExpr, children.items);
    }

    fn parseStructLiteralExprNode(self: *Parser, base: green.GreenNodeId, terminators: []const green.TokenKind) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .node = base });
        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(.RightBrace)) {
            if (self.at(.Comma)) {
                try children.append(self.allocator, .{ .token = self.bump() });
                continue;
            }
            try children.append(self.allocator, .{ .node = try self.parseStructLiteralFieldNode() });
        }
        if (self.at(.RightBrace)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected '}' after struct literal");
        }
        _ = terminators;
        return self.finishNode(SyntaxKind.StructLiteral, children.items);
    }

    fn parseStructLiteralFieldNode(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (self.at(.Identifier)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected field name in struct literal");
            return self.finishNode(SyntaxKind.AnonymousStructLiteralField, children.items);
        }

        if (self.at(.Colon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ':' after struct literal field name");
            return self.finishNode(SyntaxKind.AnonymousStructLiteralField, children.items);
        }

        try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{ .Comma, .RightBrace }) });
        return self.finishNode(SyntaxKind.AnonymousStructLiteralField, children.items);
    }

    fn parseExpressionErrorNode(self: *Parser, message: []const u8) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try self.reportHere(message);
        if (!self.at(.Eof)) {
            try children.append(self.allocator, try self.parseElement(null));
        }
        return self.finishNode(SyntaxKind.ErrorExpr, children.items);
    }

    fn parseErrorItemNode(self: *Parser, top_level: bool) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        if (!self.at(.Eof)) {
            try children.append(self.allocator, try self.parseElement(null));
        }

        while (!self.at(.Eof) and !self.at(.Semicolon) and !self.at(.RightBrace)) {
            if (top_level and self.startsTopLevelItem()) break;
            try children.append(self.allocator, try self.parseElement(null));
        }

        if (self.at(.Semicolon)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        }

        try self.diagnostics.appendError("unsupported top-level item", .{
            .file_id = self.file_id,
            .range = computeRange(self, children.items),
        });
        return self.finishNode(SyntaxKind.Error, children.items);
    }

    fn startsTopLevelItem(self: *const Parser) bool {
        const kind = self.current().kind;
        return switch (kind) {
            .Contract, .Pub, .Fn, .Struct, .Bitfield, .Enum, .Log, .Error, .Const, .Ghost, .Storage, .Memory, .Tstore, .Let, .Var, .Immutable => true,
            .Comptime => self.peekKind(1) == .Const,
            else => false,
        };
    }

    fn peekKind(self: *const Parser, lookahead: usize) green.TokenKind {
        const idx = @min(self.index + lookahead, self.tokens.items.len - 1);
        return self.tokens.items[idx].kind;
    }

    fn looksLikeImportItem(self: *const Parser) bool {
        var cursor = self.index;
        var saw_equal = false;

        while (cursor < self.tokens.items.len) : (cursor += 1) {
            const kind = self.tokens.items[cursor].kind;
            if (kind == .Semicolon or kind == .Eof) break;
            if (kind == .At and cursor + 1 < self.tokens.items.len and self.tokens.items[cursor + 1].kind == .Import) {
                return true;
            }
            if (kind == .Import) return true;
            if (!saw_equal) {
                saw_equal = kind == .Equal;
                continue;
            }
        }

        return false;
    }

    fn looksLikeLabeledBlockStmt(self: *const Parser) bool {
        return self.at(.Identifier) and self.peekKind(1) == .Colon and self.peekKind(2) == .LeftBrace;
    }

    fn looksLikeDestructuringAssignStmt(self: *const Parser) bool {
        var cursor = self.index;
        while (cursor < self.tokens.items.len) : (cursor += 1) {
            const kind = self.tokens.items[cursor].kind;
            if (kind == .Semicolon or kind == .RightBrace or kind == .Eof) return false;
            if (kind == .Equal) return false;
            if (kind == .Dot and cursor + 1 < self.tokens.items.len and self.tokens.items[cursor + 1].kind == .LeftBrace) {
                return true;
            }
        }
        return false;
    }

    fn looksLikeDirectiveStmt(self: *const Parser, comptime name: []const u8) bool {
        if (!self.at(.At) or self.peekKind(1) != .Identifier) return false;
        const token = self.tokens.items[self.index + 1];
        return std.mem.eql(u8, self.source_text[token.range.start..token.range.end], name);
    }

    fn tokenIsIdentifierLike(kind: green.TokenKind) bool {
        return switch (kind) {
            .Identifier, .From, .To, .Error, .Result => true,
            else => false,
        };
    }

    fn reportHere(self: *Parser, message: []const u8) !void {
        try self.diagnostics.appendError(message, .{
            .file_id = self.file_id,
            .range = self.current().range,
        });
    }

    fn reportUnterminated(self: *Parser, message: []const u8, node_children: []const ChildRef) !void {
        const last_token = if (node_children.len > 0) childRange(self, node_children[node_children.len - 1]) else source.TextRange.empty(0);
        try self.diagnostics.appendError(message, .{
            .file_id = self.file_id,
            .range = last_token,
        });
    }

    fn parseDelimited(self: *Parser, kind: SyntaxKind, closing: green.TokenKind) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        try children.append(self.allocator, .{ .token = self.bump() });
        while (!self.at(.Eof) and !self.at(closing)) {
            try children.append(self.allocator, try self.parseElement(closing));
        }

        if (self.at(closing)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportUnterminated("unterminated delimiter group", children.items);
        }

        return self.finishNode(kind, children.items);
    }

    fn parseUnexpectedCloser(self: *Parser) anyerror!green.GreenNodeId {
        var children: std.ArrayList(ChildRef) = .{};
        defer children.deinit(self.allocator);

        const token_id = self.bump();
        try children.append(self.allocator, .{ .token = token_id });
        try self.diagnostics.appendError("unexpected closing delimiter", .{
            .file_id = self.file_id,
            .range = self.tokens.items[token_id.index()].range,
        });
        return self.finishNode(SyntaxKind.Error, children.items);
    }

    fn finishNode(self: *Parser, kind: SyntaxKind, node_children: []const ChildRef) anyerror!green.GreenNodeId {
        const start = self.children.items.len;
        try self.children.appendSlice(self.allocator, node_children);

        const range = computeRange(self, node_children);
        const id = green.GreenNodeId.fromIndex(self.nodes.items.len);
        try self.nodes.append(self.allocator, .{
            .kind = kind,
            .children_start = @intCast(start),
            .children_len = @intCast(node_children.len),
            .range = range,
        });
        return id;
    }

    fn at(self: *const Parser, kind: green.TokenKind) bool {
        return self.current().kind == kind;
    }

    fn atAny(self: *const Parser, kinds: []const green.TokenKind) bool {
        for (kinds) |kind| {
            if (self.at(kind)) return true;
        }
        return false;
    }

    fn current(self: *const Parser) green.GreenToken {
        return self.tokens.items[self.index];
    }

    fn bump(self: *Parser) green.GreenTokenId {
        const id = green.GreenTokenId.fromIndex(self.index);
        self.index += 1;
        return id;
    }

    fn nodeCouldStartStructLiteral(self: *const Parser, node_id: green.GreenNodeId) bool {
        return switch (self.nodes.items[node_id.index()].kind) {
            .NameExpr => true,
            .GroupExpr => blk: {
                const node = self.nodes.items[node_id.index()];
                var i: usize = 0;
                while (i < node.children_len) : (i += 1) {
                    const child = self.children.items[node.children_start + i];
                    switch (child) {
                        .token => {},
                        .node => |child_id| if (self.nodeCouldStartStructLiteral(child_id)) break :blk true,
                    }
                }
                break :blk false;
            },
            else => false,
        };
    }

    fn typeAtGreaterToken(self: *const Parser) bool {
        if (self.pending_type_gt > 0) return true;
        return self.at(.Greater) or self.at(.GreaterGreater);
    }

    fn bumpTypeGreater(self: *Parser) green.GreenTokenId {
        if (self.pending_type_gt > 0) {
            self.pending_type_gt -= 1;
            const original_index = if (self.index > 0) self.index - 1 else self.index;
            const original = self.tokens.items[original_index];
            const synthetic_range = source.TextRange.init(original.range.end - 1, original.range.end);
            const synthetic_id = green.GreenTokenId.fromIndex(self.tokens.items.len);
            self.tokens.append(self.allocator, .{
                .kind = .Greater,
                .range = synthetic_range,
                .leading_trivia_start = original.leading_trivia_start,
                .leading_trivia_len = 0,
                .trailing_trivia_start = original.trailing_trivia_start,
                .trailing_trivia_len = original.trailing_trivia_len,
            }) catch unreachable;
            return synthetic_id;
        }
        if (self.at(.GreaterGreater)) {
            const original_index = self.index;
            _ = self.bump();
            const original = self.tokens.items[original_index];
            const synthetic_id = green.GreenTokenId.fromIndex(self.tokens.items.len);
            self.tokens.append(self.allocator, .{
                .kind = .Greater,
                .range = source.TextRange.init(original.range.start, original.range.start + 1),
                .leading_trivia_start = original.leading_trivia_start,
                .leading_trivia_len = original.leading_trivia_len,
                .trailing_trivia_start = original.trailing_trivia_start,
                .trailing_trivia_len = 0,
            }) catch unreachable;
            self.pending_type_gt = 1;
            return synthetic_id;
        }
        return self.bump();
    }

    fn appendConditionExpr(self: *Parser, children: *std.ArrayList(ChildRef)) anyerror!void {
        try children.append(self.allocator, .{ .token = self.bump() });
        if (!self.at(.RightParen)) {
            try children.append(self.allocator, .{ .node = try self.parseExpressionNode(&.{.RightParen}) });
        }
        if (self.at(.RightParen)) {
            try children.append(self.allocator, .{ .token = self.bump() });
        } else {
            try self.reportHere("expected ')' after condition");
        }
    }
};

fn computeRange(parser: *const Parser, node_children: []const ChildRef) source.TextRange {
    if (node_children.len == 0) {
        return source.TextRange.empty(0);
    }
    const first = childRange(parser, node_children[0]);
    const last = childRange(parser, node_children[node_children.len - 1]);
    return .{ .start = first.start, .end = last.end };
}

fn childRange(parser: *const Parser, child: ChildRef) source.TextRange {
    return switch (child) {
        .token => |token_id| parser.tokens.items[token_id.index()].range,
        .node => |node_id| parser.nodes.items[node_id.index()].range,
    };
}

fn isAssignmentToken(kind: green.TokenKind) bool {
    return switch (kind) {
        .Equal, .PlusEqual, .MinusEqual, .StarEqual, .SlashEqual, .PercentEqual => true,
        else => false,
    };
}

fn binaryOpPrecedence(kind: green.TokenKind) ?u8 {
    return switch (kind) {
        .PipePipe => 10,
        .AmpersandAmpersand => 20,
        .Pipe => 30,
        .Caret => 40,
        .Ampersand => 50,
        .EqualEqual, .BangEqual => 60,
        .Less, .LessEqual, .Greater, .GreaterEqual => 70,
        .LessLess, .GreaterGreater, .LessLessPercent, .GreaterGreaterPercent => 80,
        .Plus, .Minus, .PlusPercent, .MinusPercent => 90,
        .Star, .Slash, .Percent, .StarPercent => 100,
        else => null,
    };
}

fn copyTrivia(allocator: std.mem.Allocator, trivia_slice: []const lexer.TriviaPiece) ![]green.GreenTrivia {
    const trivia = try allocator.alloc(green.GreenTrivia, trivia_slice.len);
    for (trivia_slice, 0..) |piece, index| {
        trivia[index] = .{
            .kind = piece.kind,
            .range = .{
                .start = piece.span.start_offset,
                .end = piece.span.end_offset,
            },
        };
    }
    return trivia;
}

fn copyTokens(allocator: std.mem.Allocator, token_slice: []const lexer.Token) ![]green.GreenToken {
    const tokens = try allocator.alloc(green.GreenToken, token_slice.len);
    for (token_slice, 0..) |token, index| {
        tokens[index] = .{
            .kind = token.type,
            .range = .{
                .start = token.range.start_offset,
                .end = token.range.end_offset,
            },
            .leading_trivia_start = token.leading_trivia_start,
            .leading_trivia_len = token.leading_trivia_len,
            .trailing_trivia_start = token.trailing_trivia_start,
            .trailing_trivia_len = token.trailing_trivia_len,
        };
    }
    return tokens;
}
