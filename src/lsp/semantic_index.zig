const std = @import("std");
const lexer = @import("ora_lexer");
const parser = @import("../parser.zig");
const ast = @import("ora_ast");
const type_info = @import("ora_types").type_info;
const frontend = @import("frontend.zig");

const Allocator = std.mem.Allocator;

pub const SymbolKind = enum {
    contract,
    function,
    method,
    variable,
    field,
    constant,
    parameter,
    struct_decl,
    bitfield_decl,
    enum_decl,
    enum_member,
    event,
    error_decl,
};

pub const Symbol = struct {
    name: []const u8,
    detail: ?[]const u8 = null,
    kind: SymbolKind,
    range: frontend.Range,
    selection_range: frontend.Range,
    parent: ?usize = null,
};

pub const SemanticIndex = struct {
    symbols: []Symbol,
    parse_succeeded: bool,

    pub fn deinit(self: *SemanticIndex, allocator: Allocator) void {
        for (self.symbols) |symbol| {
            allocator.free(symbol.name);
            if (symbol.detail) |detail| {
                allocator.free(detail);
            }
        }
        allocator.free(self.symbols);
    }
};

pub const DocumentSymbol = struct {
    name: []const u8,
    detail: ?[]const u8 = null,
    kind: u8,
    range: frontend.Range,
    selectionRange: frontend.Range,
    children: []DocumentSymbol = &.{},

    pub fn deinit(self: *DocumentSymbol, allocator: Allocator) void {
        for (self.children) |*child| {
            child.deinit(allocator);
        }
        allocator.free(self.children);
    }
};

pub fn deinitDocumentSymbols(allocator: Allocator, symbols: []DocumentSymbol) void {
    for (symbols) |*symbol| {
        symbol.deinit(allocator);
    }
    allocator.free(symbols);
}

pub fn indexDocument(allocator: Allocator, source: []const u8) !SemanticIndex {
    var builder = SymbolBuilder.init(allocator);
    errdefer builder.deinit();

    var lex = try lexer.Lexer.initWithConfig(allocator, source, lexer.LexerConfig.development());
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    const previous_parser_stderr = parser.diagnostics.enable_stderr_diagnostics;
    parser.diagnostics.enable_stderr_diagnostics = false;
    defer parser.diagnostics.enable_stderr_diagnostics = previous_parser_stderr;

    var parse_result = parser.parseRaw(allocator, tokens) catch {
        return .{
            .symbols = try builder.finish(),
            .parse_succeeded = false,
        };
    };
    defer parse_result.arena.deinit();

    for (parse_result.nodes) |node| {
        try collectNode(&builder, node, null, false);
    }

    return .{
        .symbols = try builder.finish(),
        .parse_succeeded = true,
    };
}

pub fn buildDocumentSymbols(allocator: Allocator, symbols: []const Symbol) ![]DocumentSymbol {
    if (symbols.len == 0) {
        return try allocator.alloc(DocumentSymbol, 0);
    }

    const child_lists = try allocator.alloc(std.ArrayList(usize), symbols.len);
    defer {
        for (child_lists) |*list| {
            list.deinit(allocator);
        }
        allocator.free(child_lists);
    }
    for (child_lists) |*list| {
        list.* = .{};
    }

    var roots = std.ArrayList(usize){};
    defer roots.deinit(allocator);

    for (symbols, 0..) |symbol, symbol_index| {
        if (symbol.parent) |parent_index| {
            if (parent_index < symbols.len) {
                try child_lists[parent_index].append(allocator, symbol_index);
            } else {
                try roots.append(allocator, symbol_index);
            }
        } else {
            try roots.append(allocator, symbol_index);
        }
    }

    const root_indices = try roots.toOwnedSlice(allocator);
    defer allocator.free(root_indices);

    const result = try allocator.alloc(DocumentSymbol, root_indices.len);
    var built: usize = 0;
    errdefer {
        for (result[0..built]) |*symbol| {
            symbol.deinit(allocator);
        }
        allocator.free(result);
    }

    for (root_indices, 0..) |root_index, i| {
        result[i] = try buildDocumentSymbolRecursive(allocator, symbols, child_lists, root_index);
        built = i + 1;
    }

    return result;
}

fn buildDocumentSymbolRecursive(
    allocator: Allocator,
    symbols: []const Symbol,
    child_lists: []const std.ArrayList(usize),
    symbol_index: usize,
) !DocumentSymbol {
    const symbol = symbols[symbol_index];
    const child_indices = child_lists[symbol_index].items;

    const children = try allocator.alloc(DocumentSymbol, child_indices.len);
    var built: usize = 0;
    errdefer {
        for (children[0..built]) |*child| {
            child.deinit(allocator);
        }
        allocator.free(children);
    }

    for (child_indices, 0..) |child_index, i| {
        children[i] = try buildDocumentSymbolRecursive(allocator, symbols, child_lists, child_index);
        built = i + 1;
    }

    return .{
        .name = symbol.name,
        .detail = symbol.detail,
        .kind = toLspKind(symbol.kind),
        .range = symbol.range,
        .selectionRange = symbol.selection_range,
        .children = children,
    };
}

fn collectNode(builder: *SymbolBuilder, node: ast.AstNode, parent: ?usize, in_contract: bool) !void {
    switch (node) {
        .Contract => |contract_decl| {
            const contract_index = try builder.addSymbol(contract_decl.name, .contract, contract_decl.span, parent, null);
            for (contract_decl.body) |member| {
                try collectNode(builder, member, contract_index, true);
            }
        },
        .Function => |function_decl| {
            const function_kind: SymbolKind = if (in_contract) .method else .function;
            const function_detail = try formatFunctionDetailAlloc(builder.allocator, function_decl);
            const function_index = try builder.addSymbol(function_decl.name, function_kind, function_decl.span, parent, function_detail);
            for (function_decl.parameters) |parameter| {
                const parameter_type = try formatTypeInfoAlloc(builder.allocator, parameter.type_info);
                _ = try builder.addSymbol(parameter.name, .parameter, parameter.span, function_index, parameter_type);
            }
        },
        .VariableDecl => |variable_decl| {
            const variable_kind: SymbolKind = if (in_contract) .field else .variable;
            const variable_type = try formatTypeInfoAlloc(builder.allocator, variable_decl.type_info);
            _ = try builder.addSymbol(variable_decl.name, variable_kind, variable_decl.span, parent, variable_type);
        },
        .Constant => |constant_decl| {
            const constant_type = try formatTypeInfoAlloc(builder.allocator, constant_decl.typ);
            _ = try builder.addSymbol(constant_decl.name, .constant, constant_decl.span, parent, constant_type);
        },
        .StructDecl => |struct_decl| {
            const struct_index = try builder.addSymbol(struct_decl.name, .struct_decl, struct_decl.span, parent, null);
            for (struct_decl.fields) |field| {
                const field_type = try formatTypeInfoAlloc(builder.allocator, field.type_info);
                _ = try builder.addSymbol(field.name, .field, field.span, struct_index, field_type);
            }
        },
        .BitfieldDecl => |bitfield_decl| {
            const bitfield_index = try builder.addSymbol(bitfield_decl.name, .bitfield_decl, bitfield_decl.span, parent, null);
            for (bitfield_decl.fields) |field| {
                const field_type = try formatTypeInfoAlloc(builder.allocator, field.type_info);
                _ = try builder.addSymbol(field.name, .field, field.span, bitfield_index, field_type);
            }
        },
        .EnumDecl => |enum_decl| {
            const enum_index = try builder.addSymbol(enum_decl.name, .enum_decl, enum_decl.span, parent, null);
            for (enum_decl.variants) |variant| {
                _ = try builder.addSymbol(variant.name, .enum_member, variant.span, enum_index, null);
            }
        },
        .LogDecl => |log_decl| {
            const log_detail = try formatLogDetailAlloc(builder.allocator, log_decl);
            const log_index = try builder.addSymbol(log_decl.name, .event, log_decl.span, parent, log_detail);
            for (log_decl.fields) |field| {
                const field_type = try formatTypeInfoAlloc(builder.allocator, field.type_info);
                _ = try builder.addSymbol(field.name, .field, field.span, log_index, field_type);
            }
        },
        .ErrorDecl => |error_decl| {
            const error_detail = try formatErrorDetailAlloc(builder.allocator, error_decl);
            const error_index = try builder.addSymbol(error_decl.name, .error_decl, error_decl.span, parent, error_detail);
            if (error_decl.parameters) |parameters| {
                for (parameters) |parameter| {
                    const parameter_type = try formatTypeInfoAlloc(builder.allocator, parameter.type_info);
                    _ = try builder.addSymbol(parameter.name, .parameter, parameter.span, error_index, parameter_type);
                }
            }
        },
        .Import => |import_decl| {
            if (import_decl.alias) |alias| {
                const detail = try std.fmt.allocPrint(builder.allocator, "import \"{s}\"", .{import_decl.path});
                _ = try builder.addSymbol(alias, .variable, import_decl.span, parent, detail);
            }
        },
        else => {},
    }
}

pub fn findSymbolAtPosition(symbols: []const Symbol, position: frontend.Position) ?usize {
    var best_index: ?usize = null;
    var best_in_selection = false;
    var best_depth: usize = 0;
    var best_span: u64 = std.math.maxInt(u64);

    for (symbols, 0..) |symbol, symbol_index| {
        const in_selection = rangeContainsPosition(symbol.selection_range, position);
        const in_range = in_selection or rangeContainsPosition(symbol.range, position);
        if (!in_range) continue;

        const depth = symbolDepth(symbols, symbol_index);
        const span = rangeSize(symbol.range);

        if (best_index == null or
            (in_selection and !best_in_selection) or
            (in_selection == best_in_selection and depth > best_depth) or
            (in_selection == best_in_selection and depth == best_depth and span < best_span))
        {
            best_index = symbol_index;
            best_in_selection = in_selection;
            best_depth = depth;
            best_span = span;
        }
    }

    return best_index;
}

fn formatFunctionDetailAlloc(allocator: Allocator, function_decl: ast.FunctionNode) ![]u8 {
    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    try writer.writeByte('(');
    for (function_decl.parameters, 0..) |parameter, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("{s}: ", .{parameter.name});

        const parameter_type = try formatTypeInfoAlloc(allocator, parameter.type_info);
        defer allocator.free(parameter_type);
        try writer.writeAll(parameter_type);
    }
    try writer.writeByte(')');

    if (function_decl.return_type_info) |return_type| {
        const return_type_text = try formatTypeInfoAlloc(allocator, return_type);
        defer allocator.free(return_type_text);
        try writer.writeAll(" -> ");
        try writer.writeAll(return_type_text);
    } else {
        try writer.writeAll(" -> void");
    }

    return buffer.toOwnedSlice(allocator);
}

fn formatErrorDetailAlloc(allocator: Allocator, error_decl: ast.Statements.ErrorDeclNode) ![]u8 {
    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    try writer.writeByte('(');
    if (error_decl.parameters) |parameters| {
        for (parameters, 0..) |parameter, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{s}: ", .{parameter.name});
            const parameter_type = try formatTypeInfoAlloc(allocator, parameter.type_info);
            defer allocator.free(parameter_type);
            try writer.writeAll(parameter_type);
        }
    }
    try writer.writeByte(')');

    return buffer.toOwnedSlice(allocator);
}

fn formatLogDetailAlloc(allocator: Allocator, log_decl: ast.LogDeclNode) ![]u8 {
    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    try writer.writeByte('(');
    for (log_decl.fields, 0..) |field, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("{s}: ", .{field.name});
        const field_type = try formatTypeInfoAlloc(allocator, field.type_info);
        defer allocator.free(field_type);
        try writer.writeAll(field_type);
    }
    try writer.writeByte(')');

    return buffer.toOwnedSlice(allocator);
}

fn formatTypeInfoAlloc(allocator: Allocator, info: type_info.TypeInfo) ![]u8 {
    if (info.ora_type) |ora_type| {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(allocator);
        try ora_type.render(buffer.writer(allocator));
        return buffer.toOwnedSlice(allocator);
    }

    return switch (info.category) {
        .Unknown => allocator.dupe(u8, "unknown"),
        else => allocator.dupe(u8, @tagName(info.category)),
    };
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

fn symbolDepth(symbols: []const Symbol, symbol_index: usize) usize {
    var depth: usize = 0;
    var current = symbols[symbol_index].parent;
    var guard: usize = 0;

    while (current) |parent_index| {
        if (parent_index >= symbols.len or guard >= symbols.len) break;
        depth += 1;
        current = symbols[parent_index].parent;
        guard += 1;
    }

    return depth;
}

fn rangeSize(range: frontend.Range) u64 {
    const line_span = if (range.end.line >= range.start.line) range.end.line - range.start.line else 0;
    const char_span = if (range.end.character >= range.start.character) range.end.character - range.start.character else 0;
    return @as(u64, line_span) * 1_000_000 + @as(u64, char_span);
}

fn spanToRange(span: ast.SourceSpan) frontend.Range {
    const start_line = if (span.line > 0) span.line - 1 else 0;
    const start_character = if (span.column > 0) span.column - 1 else 0;

    const span_len = std.math.cast(u32, span.length) orelse std.math.maxInt(u32);
    const end_character = std.math.add(u32, start_character, span_len) catch std.math.maxInt(u32);

    return .{
        .start = .{
            .line = start_line,
            .character = start_character,
        },
        .end = .{
            .line = start_line,
            .character = end_character,
        },
    };
}

fn spanToSelectionRange(span: ast.SourceSpan, name: []const u8) frontend.Range {
    var selection = spanToRange(span);
    const name_len = std.math.cast(u32, name.len) orelse std.math.maxInt(u32);
    selection.end.character = std.math.add(u32, selection.start.character, name_len) catch std.math.maxInt(u32);
    return selection;
}

fn toLspKind(kind: SymbolKind) u8 {
    return switch (kind) {
        .contract => 5, // class
        .function => 12, // function
        .method => 6, // method
        .variable => 13, // variable
        .field => 8, // field
        .constant => 14, // constant
        .parameter => 26, // typeParameter (closest stable match for declaration parameters)
        .struct_decl => 23, // struct
        .bitfield_decl => 23, // struct
        .enum_decl => 10, // enum
        .enum_member => 22, // enumMember
        .event => 24, // event
        .error_decl => 5, // class
    };
}

const SymbolBuilder = struct {
    allocator: Allocator,
    symbols: std.ArrayList(Symbol),

    fn init(allocator: Allocator) SymbolBuilder {
        return .{
            .allocator = allocator,
            .symbols = .{},
        };
    }

    fn deinit(self: *SymbolBuilder) void {
        for (self.symbols.items) |symbol| {
            self.allocator.free(symbol.name);
            if (symbol.detail) |detail| {
                self.allocator.free(detail);
            }
        }
        self.symbols.deinit(self.allocator);
    }

    fn finish(self: *SymbolBuilder) ![]Symbol {
        return self.symbols.toOwnedSlice(self.allocator);
    }

    fn addSymbol(
        self: *SymbolBuilder,
        name: []const u8,
        kind: SymbolKind,
        span: ast.SourceSpan,
        parent: ?usize,
        detail: ?[]u8,
    ) !usize {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        errdefer if (detail) |detail_text| self.allocator.free(detail_text);

        try self.symbols.append(self.allocator, .{
            .name = name_copy,
            .detail = detail,
            .kind = kind,
            .range = spanToRange(span),
            .selection_range = spanToSelectionRange(span, name),
            .parent = parent,
        });

        return self.symbols.items.len - 1;
    }
};
