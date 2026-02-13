// ============================================================================
// Ora Formatter
// ============================================================================
//
// Formats Ora source code according to canonical style rules
//
// ============================================================================

const std = @import("std");
const lib = @import("ora_lib");
const Writer = @import("writer.zig").Writer;

pub const FormatError = error{
    ParseError,
    OutOfMemory,
    WriteError,
    UnsupportedNode,
    UnsupportedType,
};

pub const FormatOptions = struct {
    line_width: u32 = 100,
    indent_size: u32 = 4,
};

pub const Formatter = struct {
    allocator: std.mem.Allocator,
    options: FormatOptions,
    writer: Writer,
    source: []const u8,
    tokens: []const lib.Token,
    trivia: []const lib.lexer.TriviaPiece,
    token_index: usize = 0, // Current token being processed
    trivia_index: usize = 0,

    pub fn init(allocator: std.mem.Allocator, source: []const u8, options: FormatOptions) Formatter {
        return Formatter{
            .allocator = allocator,
            .options = options,
            .writer = Writer.init(allocator, options.indent_size, options.line_width),
            .source = source,
            .tokens = undefined,
            .trivia = undefined,
            .token_index = 0,
            .trivia_index = 0,
        };
    }

    pub fn deinit(self: *Formatter) void {
        self.writer.deinit();
    }

    /// Format source code
    pub fn format(self: *Formatter) FormatError![]u8 {
        // Parse the source to get AST and tokens
        var lex = lib.Lexer.init(self.allocator, self.source);
        defer lex.deinit();

        self.tokens = lex.scanTokens() catch {
            return FormatError.ParseError;
        };
        defer self.allocator.free(self.tokens);
        self.trivia = lex.getTrivia();
        self.trivia_index = 0;

        // Parse tokens to AST
        var parsed = lib.parser.parseWithArena(self.allocator, self.tokens) catch {
            return FormatError.ParseError;
        };
        defer parsed.arena.deinit();
        const ast_nodes = parsed.nodes;

        // Format AST nodes
        var first = true;
        for (ast_nodes) |node| {
            if (getNodeSpan(&node)) |span| {
                try self.emitTriviaUpTo(span.byte_offset);
            }
            if (!first) {
                try self.writer.newline();
            }
            first = false;
            try self.formatNode(&node);
        }
        try self.emitTriviaUpTo(std.math.maxInt(u32));

        // Ensure trailing newline
        if (self.writer.current_line_length > 0) {
            try self.writer.newline();
        }

        return try self.writer.toOwnedSlice();
    }

    // Comment preservation is best-effort: preserve trivia in source order around nodes.

    fn formatNode(self: *Formatter, node: *const lib.AstNode) FormatError!void {
        switch (node.*) {
            .Contract => |*contract| try self.formatContract(contract),
            .Function => |*function| try self.formatFunction(function),
            .VariableDecl => |*var_decl| try self.formatVariableDecl(var_decl),
            .StructDecl => |*struct_decl| try self.formatStructDecl(struct_decl),
            .EnumDecl => |*enum_decl| try self.formatEnumDecl(enum_decl),
            .LogDecl => |*log_decl| try self.formatLogDecl(log_decl),
            .Import => |*import| try self.formatImport(import),
            .ErrorDecl => |*error_decl| try self.formatErrorDecl(error_decl),
            else => return FormatError.UnsupportedNode,
        }
    }

    fn formatContract(self: *Formatter, contract: *const lib.ContractNode) FormatError!void {
        try self.writer.write("contract ");
        try self.writer.write(contract.name);
        try self.writer.write(" {");
        try self.writer.newline();

        self.writer.indent();

        var prev_group: ?ContractMemberGroup = null;
        for (contract.body, 0..) |member, i| {
            if (getNodeSpan(&member)) |span| {
                try self.emitTriviaUpTo(span.byte_offset);
            }
            const group = contractMemberGroup(member);
            if (i != 0) {
                try self.writer.newline();
                if (prev_group != null and (prev_group.? != group or group == .Functions)) {
                    try self.writer.newline();
                }
            }
            prev_group = group;
            try self.formatNode(&member);
            if (member == .VariableDecl) {
                try self.writer.write(";");
            }
            if (getNodeSpan(&member)) |span| {
                try self.emitTriviaUpTo(span.byte_offset + span.length);
            }
        }
        if (contract.body.len > 0) {
            try self.writer.newline();
        }

        self.writer.dedent();
        try self.writer.write("}");
    }

    fn emitTriviaUpTo(self: *Formatter, offset: u32) FormatError!void {
        while (self.trivia_index < self.trivia.len) {
            const piece = self.trivia[self.trivia_index];
            if (piece.span.start_offset >= offset) {
                break;
            }
            switch (piece.kind) {
                .LineComment, .BlockComment, .DocLineComment, .DocBlockComment => {
                    try self.emitComment(piece);
                },
                else => {},
            }
            self.trivia_index += 1;
        }
    }

    fn emitComment(self: *Formatter, piece: lib.lexer.TriviaPiece) FormatError!void {
        const start: usize = @intCast(piece.span.start_offset);
        const end: usize = @intCast(piece.span.end_offset);
        if (start >= end or end > self.source.len) {
            return;
        }
        if (piece.span.start_column == 1 and self.writer.current_line_length > 0) {
            try self.writer.newline();
        }
        if (self.writer.current_line_length > 0) {
            try self.writer.space();
        }
        try self.writer.write(self.source[start..end]);

        switch (piece.kind) {
            .LineComment, .DocLineComment => try self.writer.newline(),
            else => {
                if (piece.span.end_line > piece.span.start_line or piece.span.start_column == 1) {
                    try self.writer.newline();
                }
            },
        }
    }

    const ContractMemberGroup = enum {
        Decls,
        Functions,
        Other,
    };

    fn contractMemberGroup(node: lib.AstNode) ContractMemberGroup {
        return switch (node) {
            .Function => .Functions,
            .VariableDecl, .StructDecl, .EnumDecl, .LogDecl, .Import, .ErrorDecl => .Decls,
            else => .Other,
        };
    }

    fn getNodeSpan(node: *const lib.AstNode) ?lib.ast.SourceSpan {
        return switch (node.*) {
            .Contract => |contract| contract.span,
            .Function => |func| func.span,
            .VariableDecl => |var_decl| var_decl.span,
            .StructDecl => |struct_decl| struct_decl.span,
            .EnumDecl => |enum_decl| enum_decl.span,
            .LogDecl => |log_decl| log_decl.span,
            .Import => |import| import.span,
            .ErrorDecl => |error_decl| error_decl.span,
            else => null,
        };
    }

    fn getStatementSpan(stmt: *const lib.ast.Statements.StmtNode) ?lib.ast.SourceSpan {
        return switch (stmt.*) {
            .Return => |ret| ret.span,
            .VariableDecl => |var_decl| var_decl.span,
            .Expr => |expr| getExprSpan(&expr),
            .If => |if_node| if_node.span,
            .While => |while_node| while_node.span,
            .ForLoop => |for_node| for_node.span,
            .Switch => |switch_node| switch_node.span,
            .TryBlock => |try_node| try_node.span,
            .Log => |log_stmt| log_stmt.span,
            .Lock => |lock_stmt| lock_stmt.span,
            .Unlock => |unlock_stmt| unlock_stmt.span,
            .Invariant => |inv| inv.span,
            .Requires => |req| req.span,
            .Ensures => |ens| ens.span,
            .Assume => |assume| assume.span,
            .Havoc => |havoc| havoc.span,
            .ErrorDecl => |error_decl| error_decl.span,
            .LabeledBlock => |labeled| labeled.span,
            .DestructuringAssignment => |destructure| destructure.span,
            .CompoundAssignment => |compound| compound.span,
            .Break => |break_node| break_node.span,
            .Continue => |cont| cont.span,
            .Assert => |assert| assert.span,
        };
    }

    fn getExprSpan(expr: *const lib.ExprNode) ?lib.ast.SourceSpan {
        return switch (expr.*) {
            .Identifier => |id| id.span,
            .Literal => |lit| getLiteralSpan(&lit),
            .Binary => |bin| bin.span,
            .Unary => |unary| unary.span,
            .Assignment => |assign| assign.span,
            .CompoundAssignment => |compound| compound.span,
            .Call => |call| call.span,
            .Index => |index| index.span,
            .FieldAccess => |field| field.span,
            .Cast => |cast| cast.span,
            .Comptime => |ct| ct.span,
            .Old => |old| old.span,
            .Quantified => |quant| quant.span,
            .Tuple => |tuple| tuple.span,
            .SwitchExpression => |sw| sw.span,
            .Try => |try_expr| try_expr.span,
            .ErrorReturn => |err_ret| err_ret.span,
            .ErrorCast => |err_cast| err_cast.span,
            .Shift => |shift| shift.span,
            .StructInstantiation => |inst| inst.span,
            .AnonymousStruct => |anon| anon.span,
            .Range => |range| range.span,
            .LabeledBlock => |labeled| labeled.span,
            .Destructuring => |destructure| destructure.span,
            .EnumLiteral => |enum_lit| enum_lit.span,
            .ArrayLiteral => |array_lit| array_lit.span,
        };
    }

    fn getLiteralSpan(lit: *const lib.ast.Expressions.LiteralExpr) lib.ast.SourceSpan {
        return switch (lit.*) {
            .Integer => |i| i.span,
            .String => |s| s.span,
            .Bool => |b| b.span,
            .Address => |a| a.span,
            .Hex => |h| h.span,
            .Binary => |b| b.span,
            .Character => |c| c.span,
            .Bytes => |b| b.span,
        };
    }

    fn formatFunction(self: *Formatter, function: *const lib.FunctionNode) FormatError!void {
        // Visibility modifier
        if (function.visibility == .Public) {
            try self.writer.write("pub ");
        }

        try self.writer.write("fn ");
        try self.writer.write(function.name);
        try self.writer.write("(");

        // Parameters
        for (function.parameters, 0..) |param, i| {
            if (i > 0) {
                try self.writer.write(", ");
            }
            try self.formatParameter(&param);
        }

        try self.writer.write(")");

        // Return type
        if (function.return_type_info) |return_type| {
            try self.writer.write(" -> ");
            try self.formatTypeInfo(return_type);
        }

        // Requires/Ensures clauses
        for (function.requires_clauses) |req| {
            try self.writer.space();
            try self.writer.write("requires");
            try self.formatClause(req);
        }

        for (function.ensures_clauses) |ens| {
            try self.writer.space();
            try self.writer.write("ensures");
            try self.formatClause(ens);
        }

        try self.writer.space();
        try self.writer.write("{");
        try self.writer.newline();

        self.writer.indent();
        try self.formatBlock(&function.body);
        self.writer.dedent();

        try self.writer.write("}");
    }

    fn formatParameter(self: *Formatter, param: *const lib.ast.ParameterNode) FormatError!void {
        try self.writer.write(param.name);
        try self.writer.write(": ");
        try self.formatTypeInfo(param.type_info);
    }

    fn formatTypeInfo(self: *Formatter, type_info: lib.TypeInfo) FormatError!void {
        if (type_info.ora_type) |ora_type| {
            try self.formatOraType(ora_type);
        } else {
            return FormatError.UnsupportedType;
        }
    }

    fn formatOraType(self: *Formatter, ora_type: lib.OraType) FormatError!void {
        switch (ora_type) {
            .u8 => try self.writer.write("u8"),
            .u16 => try self.writer.write("u16"),
            .u32 => try self.writer.write("u32"),
            .u64 => try self.writer.write("u64"),
            .u128 => try self.writer.write("u128"),
            .u256 => try self.writer.write("u256"),
            .i8 => try self.writer.write("i8"),
            .i16 => try self.writer.write("i16"),
            .i32 => try self.writer.write("i32"),
            .i64 => try self.writer.write("i64"),
            .i128 => try self.writer.write("i128"),
            .i256 => try self.writer.write("i256"),
            .bool => try self.writer.write("bool"),
            .address => try self.writer.write("address"),
            .string => try self.writer.write("string"),
            .bytes => try self.writer.write("bytes"),
            .void => try self.writer.write("void"),
            .non_zero_address => try self.writer.write("NonZeroAddress"),
            .map => |map| {
                try self.writer.write("map<");
                try self.formatOraType(map.key.*);
                try self.writer.write(", ");
                try self.formatOraType(map.value.*);
                try self.writer.write(">");
            },
            .array => |arr| {
                try self.writer.write("[");
                try self.formatOraType(arr.elem.*);
                try self.writer.write("; ");
                try self.writeInt(arr.len);
                try self.writer.write("]");
            },
            .slice => |elem| {
                try self.writer.write("slice[");
                try self.formatOraType(elem.*);
                try self.writer.write("]");
            },
            .tuple => |elements| {
                try self.writer.write("(");
                for (elements, 0..) |elem, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.formatOraType(elem);
                }
                try self.writer.write(")");
            },
            .function => |func| {
                try self.writer.write("fn(");
                for (func.params, 0..) |param, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.formatOraType(param);
                }
                try self.writer.write(")");
                if (func.return_type) |ret| {
                    try self.writer.write(" -> ");
                    try self.formatOraType(ret.*);
                }
            },
            .error_union => |ok_type| {
                try self.writer.write("!");
                try self.formatOraType(ok_type.*);
            },
            ._union => |types| {
                for (types, 0..) |t, i| {
                    if (i > 0) try self.writer.write(" | ");
                    try self.formatOraType(t);
                }
            },
            .anonymous_struct => |fields| {
                try self.writer.write("struct { ");
                for (fields, 0..) |field, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writer.write(field.name);
                    try self.writer.write(": ");
                    try self.formatOraType(field.typ.*);
                }
                try self.writer.write(" }");
            },
            .struct_type => |name| try self.writer.write(name),
            .enum_type => |name| try self.writer.write(name),
            .contract_type => |name| try self.writer.write(name),
            .module => |name| {
                if (name) |module_name| {
                    try self.writer.write(module_name);
                } else {
                    return FormatError.UnsupportedType;
                }
            },
            .min_value => |min| {
                try self.writer.write("MinValue<");
                try self.formatOraType(min.base.*);
                try self.writer.write(", ");
                try self.writeInt(min.min);
                try self.writer.write(">");
            },
            .max_value => |max| {
                try self.writer.write("MaxValue<");
                try self.formatOraType(max.base.*);
                try self.writer.write(", ");
                try self.writeInt(max.max);
                try self.writer.write(">");
            },
            .in_range => |range| {
                try self.writer.write("InRange<");
                try self.formatOraType(range.base.*);
                try self.writer.write(", ");
                try self.writeInt(range.min);
                try self.writer.write(", ");
                try self.writeInt(range.max);
                try self.writer.write(">");
            },
            .scaled => |scaled| {
                try self.writer.write("Scaled<");
                try self.formatOraType(scaled.base.*);
                try self.writer.write(", ");
                try self.writeInt(scaled.decimals);
                try self.writer.write(">");
            },
            .exact => |base| {
                try self.writer.write("Exact<");
                try self.formatOraType(base.*);
                try self.writer.write(">");
            },
        }
    }

    fn formatClause(self: *Formatter, clause: *lib.ExprNode) FormatError!void {
        try self.writer.write("(");
        try self.formatExpression(clause);
        try self.writer.write(")");
    }

    fn formatBlock(self: *Formatter, block: *const lib.ast.Statements.BlockNode) FormatError!void {
        for (block.statements) |*stmt| {
            if (getStatementSpan(stmt)) |span| {
                try self.emitTriviaUpTo(span.byte_offset);
            }
            try self.formatStatement(stmt);
            if (self.statementNeedsSemicolon(stmt)) {
                try self.writer.write(";");
            }
            if (getStatementSpan(stmt)) |span| {
                try self.emitTriviaUpTo(span.byte_offset + span.length);
            }
            try self.writer.newline();
        }
    }

    fn statementNeedsSemicolon(self: *Formatter, stmt: *const lib.ast.Statements.StmtNode) bool {
        _ = self;
        return switch (stmt.*) {
            .If,
            .While,
            .ForLoop,
            .Switch,
            .TryBlock,
            .LabeledBlock,
            .Requires,
            .Ensures,
            .Invariant,
            .ErrorDecl,
            => false,
            else => true,
        };
    }

    fn formatStatement(self: *Formatter, stmt: *lib.ast.Statements.StmtNode) FormatError!void {
        switch (stmt.*) {
            .Return => |*ret| {
                try self.writer.write("return");
                if (ret.value) |value| {
                    try self.writer.space();
                    try self.formatExpression(&value);
                }
            },
            .VariableDecl => |*var_decl| try self.formatVariableDecl(var_decl),
            .Expr => |expr| try self.formatExpression(&expr),
            .If => |*if_node| try self.formatIf(if_node),
            .While => |*while_node| try self.formatWhile(while_node),
            .ForLoop => |*for_node| try self.formatFor(for_node),
            .Switch => |*switch_node| try self.formatSwitch(switch_node),
            .TryBlock => |*try_node| try self.formatTryBlock(try_node),
            .Log => |*log_stmt| {
                try self.writer.write("log ");
                try self.writer.write(log_stmt.event_name);
                try self.writer.write("(");
                for (log_stmt.args, 0..) |*arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.formatExpression(arg);
                }
                try self.writer.write(")");
            },
            .Lock => |*lock_stmt| {
                try self.writer.write("@lock(");
                try self.formatExpression(&lock_stmt.path);
                try self.writer.write(")");
            },
            .Unlock => |*unlock_stmt| {
                try self.writer.write("@unlock(");
                try self.formatExpression(&unlock_stmt.path);
                try self.writer.write(")");
            },
            .Invariant => |*inv| {
                try self.writer.write("invariant(");
                try self.formatExpression(&inv.condition);
                try self.writer.write(")");
            },
            .Requires => |*req| {
                try self.writer.write("requires(");
                try self.formatExpression(&req.condition);
                try self.writer.write(")");
            },
            .Ensures => |*ens| {
                try self.writer.write("ensures(");
                try self.formatExpression(&ens.condition);
                try self.writer.write(")");
            },
            .Assume => |*assume| {
                try self.writer.write("assume(");
                try self.formatExpression(&assume.condition);
                try self.writer.write(")");
            },
            .Havoc => |*havoc| {
                try self.writer.write("havoc ");
                try self.writer.write(havoc.variable_name);
            },
            .ErrorDecl => |*error_decl| try self.formatErrorDecl(error_decl),
            .LabeledBlock => |*labeled| {
                try self.writer.write(labeled.label);
                try self.writer.write(": {");
                try self.writer.newline();
                self.writer.indent();
                try self.formatBlock(&labeled.block);
                self.writer.dedent();
                try self.writer.write("}");
            },
            .DestructuringAssignment => |*destructure| {
                try self.writer.write("let ");
                try self.formatDestructuringPattern(&destructure.pattern);
                try self.writer.write(" = ");
                try self.formatExpression(destructure.value);
            },
            .CompoundAssignment => |*compound| {
                try self.formatExpression(compound.target);
                try self.writer.space();
                try self.formatCompoundOp(compound.operator);
                try self.writer.space();
                try self.formatExpression(compound.value);
            },
            .Break => |*break_node| {
                try self.writer.write("break");
                if (break_node.label) |label| {
                    try self.writer.write(" :");
                    try self.writer.write(label);
                }
                if (break_node.value) |value| {
                    try self.writer.space();
                    try self.formatExpression(value);
                }
            },
            .Continue => |*cont| {
                try self.writer.write("continue");
                if (cont.label) |label| {
                    try self.writer.write(" :");
                    try self.writer.write(label);
                }
            },
            .Assert => |*assert| {
                try self.writer.write("assert(");
                try self.formatExpression(&assert.condition);
                if (assert.message) |msg| {
                    try self.writer.write(", ");
                    try self.writer.write("\"");
                    try self.writeEscapedString(msg);
                    try self.writer.write("\"");
                }
                try self.writer.write(")");
            },
        }
    }

    fn formatIf(self: *Formatter, if_node: *const lib.ast.Statements.IfNode) FormatError!void {
        try self.writer.write("if (");
        try self.formatExpression(&if_node.condition);
        try self.writer.write(") {");
        try self.writer.newline();

        self.writer.indent();
        try self.formatBlock(&if_node.then_branch);
        self.writer.dedent();

        try self.writer.write("}");
        if (if_node.else_branch) |*else_branch| {
            try self.writer.write(" else {");
            try self.writer.newline();

            self.writer.indent();
            try self.formatBlock(else_branch);
            self.writer.dedent();

            try self.writer.write("}");
        }
    }

    fn formatWhile(self: *Formatter, while_node: *const lib.ast.Statements.WhileNode) FormatError!void {
        try self.writer.write("while (");
        try self.formatExpression(&while_node.condition);
        try self.writer.write(")");

        // Format invariants
        for (while_node.invariants) |*invariant| {
            try self.writer.newline();
            self.writer.indent();
            try self.writer.write("invariant(");
            try self.formatExpression(invariant);
            try self.writer.write(")");
            self.writer.dedent();
        }

        if (while_node.decreases) |decreases| {
            try self.writer.newline();
            self.writer.indent();
            try self.writer.write("decreases ");
            try self.formatExpression(decreases);
            self.writer.dedent();
        }

        try self.writer.space();
        try self.writer.write("{");
        try self.writer.newline();

        self.writer.indent();
        try self.formatBlock(&while_node.body);
        self.writer.dedent();

        try self.writer.write("}");
    }

    fn formatFor(self: *Formatter, for_node: *const lib.ast.Statements.ForLoopNode) FormatError!void {
        try self.writer.write("for (");
        try self.formatExpression(&for_node.iterable);
        try self.writer.write(") ");

        // Format loop pattern
        switch (for_node.pattern) {
            .Single => |single| {
                try self.writer.write("|");
                try self.writer.write(single.name);
                try self.writer.write("|");
            },
            .IndexPair => |pair| {
                try self.writer.write("|");
                try self.writer.write(pair.item);
                try self.writer.write(", ");
                try self.writer.write(pair.index);
                try self.writer.write("|");
            },
            .Destructured => |_| {
                try self.writer.write("|");
                // Format destructuring pattern
                try self.writer.write(".");
                try self.writer.write("{");
                // TODO: Format destructuring fields
                try self.writer.write("}");
                try self.writer.write("|");
            },
        }

        // Format invariants
        for (for_node.invariants) |*invariant| {
            try self.writer.newline();
            self.writer.indent();
            try self.writer.write("invariant(");
            try self.formatExpression(invariant);
            try self.writer.write(")");
            self.writer.dedent();
        }

        try self.writer.space();
        try self.writer.write("{");
        try self.writer.newline();

        self.writer.indent();
        try self.formatBlock(&for_node.body);
        self.writer.dedent();

        try self.writer.write("}");
    }

    fn formatSwitch(self: *Formatter, switch_node: *const lib.ast.Statements.SwitchNode) FormatError!void {
        try self.writer.write("switch (");
        try self.formatExpression(&switch_node.condition);
        try self.writer.write(") {");
        try self.writer.newline();

        self.writer.indent();
        for (switch_node.cases) |*case| {
            try self.formatSwitchCase(case);
        }
        if (switch_node.default_case) |*default_case| {
            try self.writer.write("else => {");
            try self.writer.newline();
            self.writer.indent();
            try self.formatBlock(default_case);
            self.writer.dedent();
            try self.writer.write("},");
            try self.writer.newline();
        }
        self.writer.dedent();

        try self.writer.write("}");
    }

    fn formatSwitchCase(self: *Formatter, case: *const lib.ast.Expressions.SwitchCase) FormatError!void {
        // Format pattern
        switch (case.pattern) {
            .Literal => |*lit| {
                try self.writer.write(".");
                try self.formatLiteral(&lit.value);
            },
            .EnumValue => |*enum_val| {
                try self.writer.write(".");
                try self.writer.write(enum_val.enum_name);
                try self.writer.write(".");
                try self.writer.write(enum_val.variant_name);
            },
            .Range => |*range| {
                try self.formatExpression(range.start);
                try self.writer.write(if (range.inclusive) "..." else "..");
                try self.formatExpression(range.end);
            },
            .Else => {
                try self.writer.write("else");
            },
        }

        try self.writer.write(" => ");

        // Format body
        switch (case.body) {
            .Expression => |expr| {
                try self.formatExpression(expr);
                try self.writer.write(",");
            },
            .Block => |*block| {
                try self.writer.write("{");
                try self.writer.newline();
                self.writer.indent();
                try self.formatBlock(block);
                self.writer.dedent();
                try self.writer.write("},");
            },
            .LabeledBlock => |*labeled| {
                try self.writer.write(labeled.label);
                try self.writer.write(": {");
                try self.writer.newline();
                self.writer.indent();
                try self.formatBlock(&labeled.block);
                self.writer.dedent();
                try self.writer.write("},");
            },
        }
        try self.writer.newline();
    }

    fn formatTryBlock(self: *Formatter, try_node: *const lib.ast.Statements.TryBlockNode) FormatError!void {
        try self.writer.write("try {");
        try self.writer.newline();

        self.writer.indent();
        try self.formatBlock(&try_node.try_block);
        self.writer.dedent();

        try self.writer.write("}");
        if (try_node.catch_block) |*catch_block| {
            try self.writer.write(" catch");
            if (catch_block.error_variable) |err_var| {
                try self.writer.write(" (");
                try self.writer.write(err_var);
                try self.writer.write(")");
            }
            try self.writer.write(" {");
            try self.writer.newline();

            self.writer.indent();
            try self.formatBlock(&catch_block.block);
            self.writer.dedent();

            try self.writer.write("}");
        }
    }

    fn formatExpression(self: *Formatter, expr: *const lib.ExprNode) FormatError!void {
        switch (expr.*) {
            .Literal => |lit| try self.formatLiteral(&lit),
            .Identifier => |*id| try self.writer.write(id.name),
            .Binary => |*bin| {
                try self.formatExpression(bin.lhs);
                try self.writer.space();
                try self.formatBinaryOp(bin.operator);
                try self.writer.space();
                try self.formatExpression(bin.rhs);
            },
            .Call => |*call| {
                try self.formatExpression(call.callee);
                try self.writer.write("(");
                for (call.arguments, 0..) |arg, i| {
                    if (i > 0) {
                        try self.writer.write(", ");
                    }
                    try self.formatExpression(arg);
                }
                try self.writer.write(")");
            },
            .Index => |*idx| {
                try self.formatExpression(idx.target);
                try self.writer.write("[");
                try self.formatExpression(idx.index);
                try self.writer.write("]");
            },
            .FieldAccess => |*field| {
                try self.formatExpression(field.target);
                try self.writer.write(".");
                try self.writer.write(field.field);
            },
            .Unary => |*unary| {
                try self.formatUnaryOp(unary.operator);
                try self.formatExpression(unary.operand);
            },
            .Assignment => |*assign| {
                try self.formatExpression(assign.target);
                try self.writer.space();
                try self.writer.write("=");
                try self.writer.space();
                try self.formatExpression(assign.value);
            },
            .CompoundAssignment => |*compound| {
                try self.formatExpression(compound.target);
                try self.writer.space();
                try self.formatCompoundOp(compound.operator);
                try self.writer.space();
                try self.formatExpression(compound.value);
            },
            .Cast => |*cast| {
                try self.writer.write("@cast(");
                try self.formatTypeInfo(cast.target_type);
                try self.writer.write(", ");
                try self.formatExpression(cast.operand);
                try self.writer.write(")");
            },
            .Comptime => |*comptime_expr| {
                try self.writer.write("comptime ");
                try self.writer.write("{");
                try self.writer.newline();
                self.writer.indent();
                try self.formatBlock(&comptime_expr.block);
                self.writer.dedent();
                try self.writer.write("}");
            },
            .Old => |*old_expr| {
                try self.writer.write("old(");
                try self.formatExpression(old_expr.expr);
                try self.writer.write(")");
            },
            .Quantified => |*quant| {
                try self.writer.write(if (quant.quantifier == .Forall) "forall " else "exists ");
                try self.writer.write(quant.variable);
                try self.writer.write(": ");
                try self.formatTypeInfo(quant.variable_type);
                if (quant.condition) |cond| {
                    try self.writer.write(" where ");
                    try self.formatExpression(cond);
                }
                try self.writer.write(" => ");
                try self.formatExpression(quant.body);
            },
            .Tuple => |*tuple| {
                try self.writer.write("(");
                for (tuple.elements, 0..) |elem, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.formatExpression(elem);
                }
                if (tuple.elements.len == 1) {
                    try self.writer.write(",");
                }
                try self.writer.write(")");
            },
            .Try => |*try_expr| {
                try self.writer.write("try ");
                try self.formatExpression(try_expr.expr);
            },
            .ErrorReturn => |*err_ret| {
                try self.writer.write("error.");
                try self.writer.write(err_ret.error_name);
                if (err_ret.parameters) |params| {
                    try self.writer.write("(");
                    for (params, 0..) |param, i| {
                        if (i > 0) try self.writer.write(", ");
                        try self.formatExpression(param);
                    }
                    try self.writer.write(")");
                }
            },
            .ErrorCast => |*err_cast| {
                try self.formatExpression(err_cast.operand);
                try self.writer.write(" as ");
                try self.formatTypeInfo(err_cast.target_type);
            },
            .StructInstantiation => |*inst| {
                try self.formatExpression(inst.struct_name);
                try self.writer.write(" {");
                if (inst.fields.len > 0) {
                    try self.writer.space();
                    for (inst.fields, 0..) |field, i| {
                        if (i > 0) try self.writer.write(", ");
                        try self.writer.write(field.name);
                        try self.writer.write(": ");
                        try self.formatExpression(field.value);
                    }
                    try self.writer.space();
                }
                try self.writer.write("}");
            },
            .AnonymousStruct => |*anon| {
                try self.writer.write(".{");
                if (anon.fields.len > 0) {
                    try self.writer.space();
                    for (anon.fields, 0..) |field, i| {
                        if (i > 0) try self.writer.write(", ");
                        try self.writer.write(field.name);
                        try self.writer.write(": ");
                        try self.formatExpression(field.value);
                    }
                    try self.writer.space();
                }
                try self.writer.write("}");
            },
            .Range => |*range| {
                try self.formatExpression(range.start);
                try self.writer.write(if (range.inclusive) "..." else "..");
                try self.formatExpression(range.end);
            },
            .LabeledBlock => |*labeled| {
                try self.writer.write(labeled.label);
                try self.writer.write(": {");
                try self.writer.newline();
                self.writer.indent();
                try self.formatBlock(&labeled.block);
                self.writer.dedent();
                try self.writer.write("}");
            },
            .Destructuring => |*destructure| {
                try self.formatDestructuringPattern(&destructure.pattern);
                try self.writer.write(" = ");
                try self.formatExpression(destructure.value);
            },
            .EnumLiteral => |*enum_lit| {
                try self.writer.write(enum_lit.enum_name);
                try self.writer.write(".");
                try self.writer.write(enum_lit.variant_name);
            },
            .ArrayLiteral => |*arr| {
                try self.writer.write("[");
                for (arr.elements, 0..) |elem, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.formatExpression(elem);
                }
                try self.writer.write("]");
            },
            .SwitchExpression => |*switch_expr| {
                try self.writer.write("switch (");
                try self.formatExpression(switch_expr.condition);
                try self.writer.write(") {");
                try self.writer.newline();
                self.writer.indent();
                for (switch_expr.cases) |*case| {
                    try self.formatSwitchCase(case);
                }
                if (switch_expr.default_case) |*default_case| {
                    try self.writer.write("else => {");
                    try self.writer.newline();
                    self.writer.indent();
                    try self.formatBlock(default_case);
                    self.writer.dedent();
                    try self.writer.write("},");
                    try self.writer.newline();
                }
                self.writer.dedent();
                try self.writer.write("}");
            },
            else => return FormatError.UnsupportedNode,
        }
    }

    fn formatLiteral(self: *Formatter, lit: *const lib.ast.Expressions.LiteralExpr) FormatError!void {
        switch (lit.*) {
            .Integer => |*int| {
                try self.writer.write(int.value);
            },
            .String => |*str| {
                try self.writer.write("\"");
                try self.writeEscapedString(str.value);
                try self.writer.write("\"");
            },
            .Bool => |*b| {
                if (b.value) {
                    try self.writer.write("true");
                } else {
                    try self.writer.write("false");
                }
            },
            .Address => |*addr| {
                try self.writer.write(addr.value);
            },
            .Hex => |*hex| {
                try self.writer.write(hex.value);
            },
            .Binary => |*bin| {
                try self.writer.write(bin.value);
            },
            .Character => |*ch| {
                try self.writer.write("'");
                try self.writeEscapedChar(ch.value);
                try self.writer.write("'");
            },
            .Bytes => |*bytes| {
                try self.writer.write("hex\"");
                try self.writer.write(bytes.value);
                try self.writer.write("\"");
            },
        }
    }

    fn formatUnaryOp(self: *Formatter, op: lib.ast.Expressions.UnaryOp) FormatError!void {
        const op_str = switch (op) {
            .Minus => "-",
            .Bang => "!",
            .BitNot => "~",
        };
        try self.writer.write(op_str);
    }

    fn formatCompoundOp(self: *Formatter, op: lib.ast.Expressions.CompoundAssignmentOp) FormatError!void {
        const op_str = switch (op) {
            .PlusEqual => "+=",
            .MinusEqual => "-=",
            .StarEqual => "*=",
            .SlashEqual => "/=",
            .PercentEqual => "%=",
        };
        try self.writer.write(op_str);
    }

    fn formatBinaryOp(self: *Formatter, op: lib.ast.Expressions.BinaryOp) FormatError!void {
        const op_str = switch (op) {
            .Plus => "+",
            .Minus => "-",
            .Star => "*",
            .Slash => "/",
            .Percent => "%",
            .StarStar => "**",
            .EqualEqual => "==",
            .BangEqual => "!=",
            .Less => "<",
            .LessEqual => "<=",
            .Greater => ">",
            .GreaterEqual => ">=",
            .And => "&&",
            .Or => "||",
            .BitwiseAnd => "&",
            .BitwiseOr => "|",
            .BitwiseXor => "^",
            .LeftShift => "<<",
            .RightShift => ">>",
            .Comma => ",",
        };
        try self.writer.write(op_str);
    }

    fn formatVariableDecl(self: *Formatter, var_decl: *const lib.VariableDeclNode) FormatError!void {
        // Memory region
        const region_str = switch (var_decl.region) {
            .Storage => "storage",
            .Memory => "memory",
            .TStore => "tstore",
            .Stack => "",
            .Calldata => "calldata",
        };
        if (region_str.len > 0) {
            try self.writer.write(region_str);
            try self.writer.space();
        }

        // Variable kind
        const kind_str = switch (var_decl.kind) {
            .Var => "var",
            .Let => "let",
            .Const => "const",
            .Immutable => "immutable",
        };
        try self.writer.write(kind_str);
        try self.writer.space();

        // Variable name
        try self.writer.write(var_decl.name);

        // Type annotation
        if (var_decl.type_info.ora_type != null) {
            try self.writer.write(": ");
            try self.formatTypeInfo(var_decl.type_info);
        }

        // Initializer
        if (var_decl.value) |value| {
            try self.writer.space();
            try self.writer.write("= ");
            try self.formatExpression(value);
        }
    }

    fn formatStructDecl(self: *Formatter, struct_decl: *const lib.ast.StructDeclNode) FormatError!void {
        try self.writer.write("struct ");
        try self.writer.write(struct_decl.name);
        try self.writer.write(" {");
        try self.writer.newline();

        self.writer.indent();
        for (struct_decl.fields) |field| {
            try self.writer.write(field.name);
            try self.writer.write(": ");
            try self.formatTypeInfo(field.type_info);
            try self.writer.write(";");
            try self.writer.newline();
        }
        self.writer.dedent();

        try self.writer.write("}");
    }

    fn formatEnumDecl(self: *Formatter, enum_decl: *const lib.ast.EnumDeclNode) FormatError!void {
        try self.writer.write("enum ");
        try self.writer.write(enum_decl.name);

        if (enum_decl.underlying_type_info) |underlying| {
            try self.writer.write(": ");
            try self.formatTypeInfo(underlying);
        }

        try self.writer.write(" {");
        try self.writer.newline();

        self.writer.indent();
        for (enum_decl.variants, 0..) |variant, i| {
            if (i > 0) {
                try self.writer.write(",");
                try self.writer.newline();
            }
            try self.writer.write(variant.name);
            if (variant.value) |value| {
                try self.writer.write(" = ");
                try self.formatExpression(&value);
            }
        }
        try self.writer.newline();
        self.writer.dedent();

        try self.writer.write("}");
    }

    fn formatLogDecl(self: *Formatter, log_decl: *const lib.ast.LogDeclNode) FormatError!void {
        try self.writer.write("log ");
        try self.writer.write(log_decl.name);
        try self.writer.write("(");

        for (log_decl.fields, 0..) |field, i| {
            if (i > 0) {
                try self.writer.write(", ");
            }
            if (field.indexed) {
                try self.writer.write("indexed ");
            }
            try self.writer.write(field.name);
            try self.writer.write(": ");
            try self.formatTypeInfo(field.type_info);
        }

        try self.writer.write(");");
    }

    fn formatImport(self: *Formatter, import: *const lib.ast.ImportNode) FormatError!void {
        if (import.alias) |alias| {
            try self.writer.write("const ");
            try self.writer.write(alias);
            try self.writer.write(" = ");
        }
        try self.writer.write("@import(");
        try self.writer.write("\"");
        try self.writer.write(import.path);
        try self.writer.write("\"");
        try self.writer.write(");");
    }

    fn formatErrorDecl(self: *Formatter, error_decl: *const lib.ast.Statements.ErrorDeclNode) FormatError!void {
        try self.writer.write("error ");
        try self.writer.write(error_decl.name);

        if (error_decl.parameters) |params| {
            if (params.len > 0) {
                try self.writer.write("(");
                for (params, 0..) |param, i| {
                    if (i > 0) {
                        try self.writer.write(", ");
                    }
                    try self.formatParameter(&param);
                }
                try self.writer.write(")");
            }
        }

        try self.writer.write(";");
    }

    fn writeEscapedString(self: *Formatter, value: []const u8) FormatError!void {
        for (value) |c| {
            try self.writeEscapedChar(c);
        }
    }

    fn writeEscapedChar(self: *Formatter, value: u8) FormatError!void {
        switch (value) {
            '\n' => try self.writer.write("\\n"),
            '\r' => try self.writer.write("\\r"),
            '\t' => try self.writer.write("\\t"),
            '\\' => try self.writer.write("\\\\"),
            '"' => try self.writer.write("\\\""),
            '\'' => try self.writer.write("\\'"),
            else => {
                if (value < 0x20 or value >= 0x7f) {
                    var buf: [4]u8 = undefined;
                    const rendered = std.fmt.bufPrint(&buf, "\\x{X:0>2}", .{value}) catch return FormatError.WriteError;
                    try self.writer.write(rendered);
                } else {
                    try self.writer.writeByte(value);
                }
            },
        }
    }

    fn writeInt(self: *Formatter, value: anytype) FormatError!void {
        var buf: [128]u8 = undefined;
        const rendered = std.fmt.bufPrint(&buf, "{}", .{value}) catch return FormatError.WriteError;
        try self.writer.write(rendered);
    }

    fn formatDestructuringPattern(self: *Formatter, pattern: *const lib.ast.Expressions.DestructuringPattern) FormatError!void {
        switch (pattern.*) {
            .Struct => |fields| {
                try self.writer.write(".{");
                if (fields.len > 0) {
                    try self.writer.space();
                    for (fields, 0..) |field, i| {
                        if (i > 0) try self.writer.write(", ");
                        try self.writer.write(field.name);
                        if (!std.mem.eql(u8, field.name, field.variable)) {
                            try self.writer.write(": ");
                            try self.writer.write(field.variable);
                        }
                    }
                    try self.writer.space();
                }
                try self.writer.write("}");
            },
            .Tuple => |names| {
                try self.writer.write("(");
                for (names, 0..) |name, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writer.write(name);
                }
                try self.writer.write(")");
            },
            .Array => |names| {
                try self.writer.write("[");
                for (names, 0..) |name, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writer.write(name);
                }
                try self.writer.write("]");
            },
        }
    }
};
