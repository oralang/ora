const std = @import("std");
const compiler = @import("../compiler/mod.zig");
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
            if (symbol.detail) |detail| allocator.free(detail);
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
        for (self.children) |*child| child.deinit(allocator);
        allocator.free(self.children);
    }
};

pub fn deinitDocumentSymbols(allocator: Allocator, symbols: []DocumentSymbol) void {
    for (symbols) |*symbol| symbol.deinit(allocator);
    allocator.free(symbols);
}

pub fn indexDocument(allocator: Allocator, source: []const u8) !SemanticIndex {
    var builder = try SymbolBuilder.init(allocator, source);
    errdefer builder.deinit();

    var parse_result = try compiler.syntax.parse(allocator, compiler.FileId.fromIndex(0), source);
    defer parse_result.deinit();

    var lower_result = try compiler.ast.lower(allocator, &parse_result.tree);
    defer lower_result.deinit();

    for (lower_result.file.root_items) |item_id| {
        try collectItem(&builder, &lower_result.file, item_id, null, false);
    }

    return .{
        .symbols = try builder.finish(),
        .parse_succeeded = parse_result.diagnostics.isEmpty() and lower_result.diagnostics.isEmpty(),
    };
}

pub fn buildDocumentSymbols(allocator: Allocator, symbols: []const Symbol) ![]DocumentSymbol {
    if (symbols.len == 0) return try allocator.alloc(DocumentSymbol, 0);

    const child_lists = try allocator.alloc(std.ArrayList(usize), symbols.len);
    defer {
        for (child_lists) |*list| list.deinit(allocator);
        allocator.free(child_lists);
    }
    for (child_lists) |*list| list.* = .{};

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
        for (result[0..built]) |*symbol| symbol.deinit(allocator);
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
        for (children[0..built]) |*child| child.deinit(allocator);
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

fn collectItem(
    builder: *SymbolBuilder,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    parent: ?usize,
    in_contract: bool,
) !void {
    const item = file.item(item_id).*;
    switch (item) {
        .Contract => |contract_decl| {
            const contract_index = try builder.addSymbol(contract_decl.name, .contract, contract_decl.range, parent, null);
            for (contract_decl.members) |member_id| {
                try collectItem(builder, file, member_id, contract_index, true);
            }
        },
        .Function => |function_decl| {
            const function_kind: SymbolKind = if (in_contract) .method else .function;
            const function_detail = try formatFunctionDetailAlloc(builder.allocator, file, function_decl);
            const function_index = try builder.addSymbol(function_decl.name, function_kind, function_decl.range, parent, function_detail);
            for (function_decl.parameters) |parameter| {
                const parameter_name = patternName(file, parameter.pattern) orelse continue;
                const parameter_type = try formatTypeExprAlloc(builder.allocator, file, parameter.type_expr);
                _ = try builder.addSymbol(parameter_name, .parameter, parameter.range, function_index, parameter_type);
            }
        },
        .Field => |field_decl| {
            const variable_kind: SymbolKind = if (in_contract) .field else .variable;
            const variable_type = if (field_decl.type_expr) |type_expr|
                try formatTypeExprAlloc(builder.allocator, file, type_expr)
            else
                null;
            _ = try builder.addSymbol(field_decl.name, variable_kind, field_decl.range, parent, variable_type);
        },
        .Constant => |constant_decl| {
            const constant_type = if (constant_decl.type_expr) |type_expr|
                try formatTypeExprAlloc(builder.allocator, file, type_expr)
            else
                null;
            _ = try builder.addSymbol(constant_decl.name, .constant, constant_decl.range, parent, constant_type);
        },
        .Struct => |struct_decl| {
            const struct_index = try builder.addSymbol(struct_decl.name, .struct_decl, struct_decl.range, parent, null);
            for (struct_decl.fields) |field| {
                const field_type = try formatTypeExprAlloc(builder.allocator, file, field.type_expr);
                _ = try builder.addSymbol(field.name, .field, field.range, struct_index, field_type);
            }
        },
        .Bitfield => |bitfield_decl| {
            const bitfield_index = try builder.addSymbol(bitfield_decl.name, .bitfield_decl, bitfield_decl.range, parent, null);
            for (bitfield_decl.fields) |field| {
                const field_type = try formatTypeExprAlloc(builder.allocator, file, field.type_expr);
                _ = try builder.addSymbol(field.name, .field, field.range, bitfield_index, field_type);
            }
        },
        .Enum => |enum_decl| {
            const enum_index = try builder.addSymbol(enum_decl.name, .enum_decl, enum_decl.range, parent, null);
            for (enum_decl.variants) |variant| {
                _ = try builder.addSymbol(variant.name, .enum_member, variant.range, enum_index, null);
            }
        },
        .LogDecl => |log_decl| {
            const log_detail = try formatLogDetailAlloc(builder.allocator, file, log_decl);
            const log_index = try builder.addSymbol(log_decl.name, .event, log_decl.range, parent, log_detail);
            for (log_decl.fields) |field| {
                const field_type = try formatTypeExprAlloc(builder.allocator, file, field.type_expr);
                _ = try builder.addSymbol(field.name, .field, field.range, log_index, field_type);
            }
        },
        .ErrorDecl => |error_decl| {
            const error_detail = try formatErrorDetailAlloc(builder.allocator, file, error_decl);
            const error_index = try builder.addSymbol(error_decl.name, .error_decl, error_decl.range, parent, error_detail);
            for (error_decl.parameters) |parameter| {
                const parameter_name = patternName(file, parameter.pattern) orelse continue;
                const parameter_type = try formatTypeExprAlloc(builder.allocator, file, parameter.type_expr);
                _ = try builder.addSymbol(parameter_name, .parameter, parameter.range, error_index, parameter_type);
            }
        },
        .Import => |import_decl| {
            if (import_decl.alias) |alias| {
                const detail = try std.fmt.allocPrint(builder.allocator, "import \"{s}\"", .{import_decl.path});
                _ = try builder.addSymbol(alias, .variable, import_decl.range, parent, detail);
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

fn formatFunctionDetailAlloc(allocator: Allocator, file: *const compiler.ast.AstFile, function_decl: compiler.ast.FunctionItem) ![]u8 {
    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    try writer.writeByte('(');
    for (function_decl.parameters, 0..) |parameter, i| {
        if (i > 0) try writer.writeAll(", ");
        const parameter_name = patternName(file, parameter.pattern) orelse "_";
        try writer.print("{s}: ", .{parameter_name});

        const parameter_type = try formatTypeExprAlloc(allocator, file, parameter.type_expr);
        defer allocator.free(parameter_type);
        try writer.writeAll(parameter_type);
    }
    try writer.writeByte(')');

    if (function_decl.return_type) |return_type| {
        const return_type_text = try formatTypeExprAlloc(allocator, file, return_type);
        defer allocator.free(return_type_text);
        try writer.writeAll(" -> ");
        try writer.writeAll(return_type_text);
    }

    return buffer.toOwnedSlice(allocator);
}

fn formatErrorDetailAlloc(allocator: Allocator, file: *const compiler.ast.AstFile, error_decl: compiler.ast.ErrorDeclItem) ![]u8 {
    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    try writer.writeByte('(');
    for (error_decl.parameters, 0..) |parameter, i| {
        if (i > 0) try writer.writeAll(", ");
        const parameter_name = patternName(file, parameter.pattern) orelse "_";
        try writer.print("{s}: ", .{parameter_name});
        const parameter_type = try formatTypeExprAlloc(allocator, file, parameter.type_expr);
        defer allocator.free(parameter_type);
        try writer.writeAll(parameter_type);
    }
    try writer.writeByte(')');

    return buffer.toOwnedSlice(allocator);
}

fn formatLogDetailAlloc(allocator: Allocator, file: *const compiler.ast.AstFile, log_decl: compiler.ast.LogDeclItem) ![]u8 {
    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    try writer.writeByte('(');
    for (log_decl.fields, 0..) |field, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("{s}: ", .{field.name});
        const field_type = try formatTypeExprAlloc(allocator, file, field.type_expr);
        defer allocator.free(field_type);
        try writer.writeAll(field_type);
    }
    try writer.writeByte(')');

    return buffer.toOwnedSlice(allocator);
}

fn formatTypeExprAlloc(allocator: Allocator, file: *const compiler.ast.AstFile, type_expr_id: compiler.ast.TypeExprId) ![]u8 {
    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    try writeTypeExpr(buffer.writer(allocator), file, type_expr_id);
    return buffer.toOwnedSlice(allocator);
}

fn writeTypeExpr(writer: anytype, file: *const compiler.ast.AstFile, type_expr_id: compiler.ast.TypeExprId) !void {
    switch (file.typeExpr(type_expr_id).*) {
        .Path => |path| try writer.writeAll(path.name),
        .Generic => |generic| {
            try writer.writeAll(generic.name);
            try writer.writeByte('<');
            for (generic.args, 0..) |arg, i| {
                if (i > 0) try writer.writeAll(", ");
                switch (arg) {
                    .Type => |nested| try writeTypeExpr(writer, file, nested),
                    .Integer => |value| try writer.writeAll(value.text),
                }
            }
            try writer.writeByte('>');
        },
        .Tuple => |tuple| {
            try writer.writeByte('(');
            for (tuple.elements, 0..) |element, i| {
                if (i > 0) try writer.writeAll(", ");
                try writeTypeExpr(writer, file, element);
            }
            try writer.writeByte(')');
        },
        .Array => |array| {
            try writer.writeByte('[');
            switch (array.size) {
                .Integer => |value| try writer.writeAll(value.text),
                .Name => |name| try writer.writeAll(name.name),
            }
            try writer.writeByte(']');
            try writeTypeExpr(writer, file, array.element);
        },
        .Slice => |slice| {
            try writer.writeAll("[]");
            try writeTypeExpr(writer, file, slice.element);
        },
        .ErrorUnion => |error_union| {
            try writer.writeByte('!');
            try writeTypeExpr(writer, file, error_union.payload);
            for (error_union.errors) |err_ty| {
                try writer.writeAll(" | ");
                try writeTypeExpr(writer, file, err_ty);
            }
        },
        .Error => try writer.writeAll("unknown"),
    }
}

fn patternName(file: *const compiler.ast.AstFile, pattern_id: compiler.ast.PatternId) ?[]const u8 {
    return switch (file.pattern(pattern_id).*) {
        .Name => |name| name.name,
        else => null,
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

fn toLspKind(kind: SymbolKind) u8 {
    return switch (kind) {
        .contract => 5,
        .function => 12,
        .method => 6,
        .variable => 13,
        .field => 8,
        .constant => 14,
        .parameter => 26,
        .struct_decl => 23,
        .bitfield_decl => 23,
        .enum_decl => 10,
        .enum_member => 22,
        .event => 24,
        .error_decl => 5,
    };
}

const SymbolBuilder = struct {
    allocator: Allocator,
    symbols: std.ArrayList(Symbol),
    sources: compiler.source.SourceStore,
    file_id: compiler.FileId,
    source_text: []const u8,

    fn init(allocator: Allocator, source_text: []const u8) !SymbolBuilder {
        var sources = compiler.source.SourceStore.init(allocator);
        errdefer sources.deinit();
        const file_id = try sources.addFile("<lsp>", source_text);
        return .{
            .allocator = allocator,
            .symbols = .{},
            .sources = sources,
            .file_id = file_id,
            .source_text = source_text,
        };
    }

    fn deinit(self: *SymbolBuilder) void {
        for (self.symbols.items) |symbol| {
            self.allocator.free(symbol.name);
            if (symbol.detail) |detail| self.allocator.free(detail);
        }
        self.symbols.deinit(self.allocator);
        self.sources.deinit();
    }

    fn finish(self: *SymbolBuilder) ![]Symbol {
        const owned = try self.symbols.toOwnedSlice(self.allocator);
        self.sources.deinit();
        self.sources = compiler.source.SourceStore.init(self.allocator);
        return owned;
    }

    fn addSymbol(
        self: *SymbolBuilder,
        name: []const u8,
        kind: SymbolKind,
        range: compiler.TextRange,
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
            .range = self.textRangeToRange(range),
            .selection_range = self.textRangeToSelectionRange(range, name),
            .parent = parent,
        });

        return self.symbols.items.len - 1;
    }

    fn textRangeToRange(self: *const SymbolBuilder, range: compiler.TextRange) frontend.Range {
        const start = self.sources.lineColumn(.{
            .file_id = self.file_id,
            .range = .{ .start = range.start, .end = range.start },
        });
        const end = self.sources.lineColumn(.{
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

    fn textRangeToSelectionRange(self: *const SymbolBuilder, range: compiler.TextRange, name: []const u8) frontend.Range {
        var name_start = range.start;
        const start: usize = @intCast(@min(range.start, self.source_text.len));
        const end: usize = @intCast(@min(range.end, self.source_text.len));
        if (start <= end and end <= self.source_text.len) {
            if (std.mem.indexOf(u8, self.source_text[start..end], name)) |relative| {
                const relative_u32 = std.math.cast(u32, relative) orelse std.math.maxInt(u32);
                name_start = std.math.add(u32, range.start, relative_u32) catch range.start;
            }
        }
        var selection = self.textRangeToRange(.{
            .start = name_start,
            .end = name_start,
        });
        const name_len = std.math.cast(u32, name.len) orelse std.math.maxInt(u32);
        selection.end.line = selection.start.line;
        selection.end.character = std.math.add(u32, selection.start.character, name_len) catch std.math.maxInt(u32);
        return selection;
    }
};
