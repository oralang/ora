const std = @import("std");
const compiler = @import("../compiler.zig");
const embedded_stdlib = @import("ora_imports").embedded_stdlib;
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");
const token_cache = @import("token_cache.zig");

const Allocator = std.mem.Allocator;

pub const ImportAlias = struct {
    alias: []const u8,
    logical_path: []const u8,
};

pub const Hover = struct {
    contents: []const u8,
    range: frontend.Range,
};

pub const Definition = struct {
    uri: []const u8,
    range: frontend.Range,
};

const Module = struct {
    logical_path: []const u8,
    resolved_path: []const u8,
    source: []const u8,
    index: semantic_index.SemanticIndex,
};

const zero_range: frontend.Range = .{
    .start = .{ .line = 0, .character = 0 },
    .end = .{ .line = 0, .character = 0 },
};

const SyntheticNamespace = struct {
    name: []const u8,
    documentation: []const u8,
};

const SyntheticMember = struct {
    namespace: []const u8,
    name: []const u8,
    detail: []const u8,
    documentation: []const u8,
};

const synthetic_namespaces = [_]SyntheticNamespace{
    .{
        .name = "msg",
        .documentation = "Message-scoped EVM environment values for the current call.",
    },
    .{
        .name = "transaction",
        .documentation = "Transaction-scoped EVM environment values. `sender` is an alias of `std.msg.sender`.",
    },
    .{
        .name = "tx",
        .documentation = "Transaction-origin environment values.",
    },
    .{
        .name = "block",
        .documentation = "Current block environment values.",
    },
};

const synthetic_members = [_]SyntheticMember{
    .{
        .namespace = "msg",
        .name = "sender",
        .detail = "address",
        .documentation = "Address of the immediate caller. This value is trusted as non-zero and can flow into `NonZeroAddress`.",
    },
    .{
        .namespace = "msg",
        .name = "value",
        .detail = "u256",
        .documentation = "Wei value sent with the current call.",
    },
    .{
        .namespace = "transaction",
        .name = "sender",
        .detail = "address",
        .documentation = "Address of the immediate caller. Alias of `std.msg.sender`; trusted as non-zero.",
    },
    .{
        .namespace = "transaction",
        .name = "gasprice",
        .detail = "u256",
        .documentation = "Gas price for the current transaction.",
    },
    .{
        .namespace = "tx",
        .name = "origin",
        .detail = "address",
        .documentation = "Original externally owned account that started the transaction.",
    },
    .{
        .namespace = "block",
        .name = "timestamp",
        .detail = "u256",
        .documentation = "Timestamp of the current block.",
    },
    .{
        .namespace = "block",
        .name = "number",
        .detail = "u256",
        .documentation = "Number of the current block.",
    },
    .{
        .namespace = "block",
        .name = "coinbase",
        .detail = "address",
        .documentation = "Address of the current block beneficiary.",
    },
};

pub const Index = struct {
    arena: std.heap.ArenaAllocator,
    modules: []Module,

    pub fn init(allocator: Allocator) !Index {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const arena_allocator = arena.allocator();

        const embedded = embedded_stdlib.all();
        const modules = try arena_allocator.alloc(Module, embedded.len);
        var built: usize = 0;
        for (embedded) |embedded_module| {
            const index = blk: {
                var db = compiler.CompilerDb.init(arena_allocator);
                defer db.deinitFrontendOnly();

                const file_id = try db.addSourceFile(embedded_module.resolved_path, embedded_module.source);
                const ast_file = try db.astFile(file_id);
                const parse_succeeded =
                    (try db.syntaxDiagnostics(file_id)).isEmpty() and
                    (try db.astDiagnostics(file_id)).isEmpty();

                break :blk try semantic_index.indexAstFileWithSourceStoreAlloc(
                    arena_allocator,
                    arena_allocator,
                    &db.sources,
                    file_id,
                    embedded_module.source,
                    ast_file,
                    parse_succeeded,
                );
            };

            modules[built] = .{
                .logical_path = embedded_module.logical_path,
                .resolved_path = embedded_module.resolved_path,
                .source = embedded_module.source,
                .index = index,
            };
            built += 1;
        }

        return .{
            .arena = arena,
            .modules = modules[0..built],
        };
    }

    pub fn deinit(self: *Index) void {
        self.arena.deinit();
        self.* = undefined;
    }

    fn module(self: *const Index, logical_path: []const u8) ?*const Module {
        for (self.modules) |*item| {
            if (std.mem.eql(u8, item.logical_path, logical_path)) return item;
        }
        return null;
    }
};

pub fn collectImportAliases(allocator: Allocator, tokens: []const token_cache.Token) ![]ImportAlias {
    var aliases = std.ArrayList(ImportAlias){};
    errdefer {
        for (aliases.items) |alias| {
            allocator.free(alias.alias);
            allocator.free(alias.logical_path);
        }
        aliases.deinit(allocator);
    }

    var index: usize = 0;
    while (index < tokens.len) : (index += 1) {
        const token = tokens[index];
        if (token.type != .Const and token.type != .Comptime) continue;

        const import_start = if (token.type == .Comptime) index + 1 else index;
        if (import_start + 6 >= tokens.len) continue;
        if (tokens[import_start].type != .Const) continue;

        const alias_token = tokens[import_start + 1];
        const equal_token = tokens[import_start + 2];
        const at_token = tokens[import_start + 3];
        const import_token = tokens[import_start + 4];
        const left_paren_token = tokens[import_start + 5];
        const path_token = tokens[import_start + 6];
        if (!isAliasToken(alias_token) or
            equal_token.type != .Equal or
            at_token.type != .At or
            import_token.type != .Import or
            left_paren_token.type != .LeftParen or
            (path_token.type != .StringLiteral and path_token.type != .RawStringLiteral))
        {
            continue;
        }

        const logical_path = path_token.string_value orelse continue;
        if (embedded_stdlib.byLogicalPath(logical_path) == null) continue;

        try aliases.append(allocator, .{
            .alias = try allocator.dupe(u8, alias_token.lexeme),
            .logical_path = try allocator.dupe(u8, logical_path),
        });
        index = import_start + 6;
    }

    return aliases.toOwnedSlice(allocator);
}

pub fn hoverAt(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    index: *const Index,
    aliases: []const ImportAlias,
) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    const chain = accessChainEndingAt(source, word) orelse return null;
    if (try syntheticHoverAtChain(allocator, aliases, chain)) |value| {
        return .{ .contents = value, .range = word.range };
    }

    const resolved = resolveChain(index, aliases, chain) orelse return null;

    const value = try formatHoverValueAlloc(allocator, resolved.symbol);
    return .{ .contents = value, .range = word.range };
}

pub fn definitionAt(
    source: []const u8,
    position: frontend.Position,
    index: *const Index,
    aliases: []const ImportAlias,
) ?Definition {
    const word = wordAtPosition(source, position) orelse return null;
    const chain = accessChainEndingAt(source, word) orelse return null;
    const resolved = resolveChain(index, aliases, chain) orelse return null;
    return .{
        .uri = resolved.module.resolved_path,
        .range = resolved.symbol.selection_range,
    };
}

pub fn completionCandidatesAt(
    allocator: Allocator,
    source: []const u8,
    cursor_offset: usize,
    prefix: []const u8,
    index: *const Index,
    aliases: []const ImportAlias,
) ![]semantic_index.Symbol {
    var symbols = std.ArrayList(semantic_index.Symbol){};
    errdefer symbols.deinit(allocator);

    const chain = completionChainForPrefix(source, cursor_offset, prefix) orelse
        return symbols.toOwnedSlice(allocator);

    if (isStdRootCompletionChain(aliases, chain)) {
        if (index.module("std")) |root_module| {
            try appendModuleCompletionCandidates(allocator, &symbols, root_module, prefix);
        }
        try appendSyntheticNamespaceCandidates(allocator, &symbols, prefix);
    } else if (stdSyntheticNamespaceCompletion(aliases, chain)) |namespace| {
        try appendSyntheticMemberCandidates(allocator, &symbols, namespace, prefix);
    } else {
        const target_module = moduleForCompletionChain(index, aliases, chain) orelse
            return symbols.toOwnedSlice(allocator);
        try appendModuleCompletionCandidates(allocator, &symbols, target_module, prefix);
    }

    std.sort.heap(semantic_index.Symbol, symbols.items, {}, lessSymbolByName);
    return symbols.toOwnedSlice(allocator);
}

pub fn completionCandidatesAtPosition(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    index: *const Index,
    aliases: []const ImportAlias,
) ![]semantic_index.Symbol {
    const cursor_offset = positionToByteOffsetOnLine(source, position);
    const prefix = identifierPrefixAtOffset(source, cursor_offset);
    return completionCandidatesAt(allocator, source, cursor_offset, prefix, index, aliases);
}

pub fn isCompletionAccessContext(source: []const u8, position: frontend.Position) bool {
    const cursor_offset = positionToByteOffsetOnLine(source, position);
    const prefix = identifierPrefixAtOffset(source, cursor_offset);
    return completionChainForPrefix(source, cursor_offset, prefix) != null;
}

const Word = struct {
    text: []const u8,
    range: frontend.Range,
    start_offset: usize,
    end_offset: usize,
};

const Chain = struct {
    segments: [4][]const u8 = undefined,
    len: usize = 0,

    fn append(self: *Chain, segment: []const u8) bool {
        if (self.len >= self.segments.len) return false;
        self.segments[self.len] = segment;
        self.len += 1;
        return true;
    }

    fn at(self: Chain, index: usize) []const u8 {
        return self.segments[index];
    }
};

const ResolvedSymbol = struct {
    module: *const Module,
    symbol: semantic_index.Symbol,
};

const ResolvedTarget = struct {
    module_path: []const u8,
    symbol_name: []const u8,
};

fn syntheticHoverAtChain(allocator: Allocator, aliases: []const ImportAlias, chain: Chain) !?[]u8 {
    if (chain.len < 2) return null;
    const first_alias = findAlias(aliases, chain.at(0)) orelse return null;
    if (!std.mem.eql(u8, first_alias.logical_path, "std")) return null;

    if (chain.len == 2) {
        const namespace = syntheticNamespace(chain.at(1)) orelse return null;
        return try std.fmt.allocPrint(
            allocator,
            "```ora\nstd.{s}\n```\n---\n{s}",
            .{ namespace.name, namespace.documentation },
        );
    }

    if (chain.len == 3) {
        const member = syntheticMember(chain.at(1), chain.at(2)) orelse return null;
        return try syntheticMemberMarkdownAlloc(allocator, member);
    }

    return null;
}

fn resolveChain(index: *const Index, aliases: []const ImportAlias, chain: Chain) ?ResolvedSymbol {
    if (chain.len < 2) return null;

    const first_alias = findAlias(aliases, chain.at(0)) orelse return null;
    const target: ResolvedTarget = if (std.mem.eql(u8, first_alias.logical_path, "std")) blk: {
        if (chain.len < 3) return null;
        const submodule = modulePathForRootMember(index, chain.at(1)) orelse return null;
        break :blk .{ .module_path = submodule, .symbol_name = chain.at(2) };
    } else .{ .module_path = first_alias.logical_path, .symbol_name = chain.at(1) };

    const module = index.module(target.module_path) orelse return null;
    const symbol = findSymbol(module.index.symbols, target.symbol_name) orelse return null;
    return .{ .module = module, .symbol = symbol };
}

fn syntheticNamespace(name: []const u8) ?SyntheticNamespace {
    for (synthetic_namespaces) |namespace| {
        if (std.mem.eql(u8, namespace.name, name)) return namespace;
    }
    return null;
}

fn syntheticMember(namespace: []const u8, name: []const u8) ?SyntheticMember {
    for (synthetic_members) |member| {
        if (std.mem.eql(u8, member.namespace, namespace) and
            std.mem.eql(u8, member.name, name))
        {
            return member;
        }
    }
    return null;
}

fn syntheticNamespaceSymbol(namespace: SyntheticNamespace) semantic_index.Symbol {
    return .{
        .name = namespace.name,
        .detail = "std namespace",
        .doc_comment = namespace.documentation,
        .kind = .variable,
        .range = zero_range,
        .selection_range = zero_range,
    };
}

fn syntheticMemberSymbol(member: SyntheticMember) semantic_index.Symbol {
    return .{
        .name = member.name,
        .detail = member.detail,
        .doc_comment = member.documentation,
        .kind = .constant,
        .range = zero_range,
        .selection_range = zero_range,
    };
}

fn syntheticMemberMarkdownAlloc(allocator: Allocator, member: SyntheticMember) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "```ora\nstd.{s}.{s}: {s}\n```\n---\n{s}",
        .{ member.namespace, member.name, member.detail, member.documentation },
    );
}

fn moduleForCompletionChain(
    index: *const Index,
    aliases: []const ImportAlias,
    chain: Chain,
) ?*const Module {
    const first_alias = findAlias(aliases, chain.at(0)) orelse return null;

    if (chain.len == 1) {
        return index.module(first_alias.logical_path);
    }

    if (chain.len == 2 and std.mem.eql(u8, first_alias.logical_path, "std")) {
        const submodule = modulePathForRootMember(index, chain.at(1)) orelse return null;
        return index.module(submodule);
    }

    return null;
}

fn completionChainForPrefix(source: []const u8, cursor_offset: usize, prefix: []const u8) ?Chain {
    if (cursor_offset < prefix.len) return null;
    const prefix_start = cursor_offset - prefix.len;
    if (prefix_start == 0 or source[prefix_start - 1] != '.') return null;
    return accessChainBeforeDot(source, prefix_start - 1);
}

fn isStdRootCompletionChain(aliases: []const ImportAlias, chain: Chain) bool {
    if (chain.len != 1) return false;
    const first_alias = findAlias(aliases, chain.at(0)) orelse return false;
    return std.mem.eql(u8, first_alias.logical_path, "std");
}

fn stdSyntheticNamespaceCompletion(aliases: []const ImportAlias, chain: Chain) ?[]const u8 {
    if (chain.len != 2) return null;
    const first_alias = findAlias(aliases, chain.at(0)) orelse return null;
    if (!std.mem.eql(u8, first_alias.logical_path, "std")) return null;
    return if (syntheticNamespace(chain.at(1)) != null) chain.at(1) else null;
}

fn appendModuleCompletionCandidates(
    allocator: Allocator,
    symbols: *std.ArrayList(semantic_index.Symbol),
    module: *const Module,
    prefix: []const u8,
) !void {
    for (module.index.symbols) |symbol| {
        if (!isExportedCompletionSymbol(symbol)) continue;
        if (!std.mem.startsWith(u8, symbol.name, prefix)) continue;
        try symbols.append(allocator, symbol);
    }
}

fn appendSyntheticNamespaceCandidates(
    allocator: Allocator,
    symbols: *std.ArrayList(semantic_index.Symbol),
    prefix: []const u8,
) !void {
    for (synthetic_namespaces) |namespace| {
        if (!std.mem.startsWith(u8, namespace.name, prefix)) continue;
        try symbols.append(allocator, syntheticNamespaceSymbol(namespace));
    }
}

fn appendSyntheticMemberCandidates(
    allocator: Allocator,
    symbols: *std.ArrayList(semantic_index.Symbol),
    namespace: []const u8,
    prefix: []const u8,
) !void {
    for (synthetic_members) |member| {
        if (!std.mem.eql(u8, member.namespace, namespace)) continue;
        if (!std.mem.startsWith(u8, member.name, prefix)) continue;
        try symbols.append(allocator, syntheticMemberSymbol(member));
    }
}

fn findAlias(aliases: []const ImportAlias, alias_name: []const u8) ?ImportAlias {
    for (aliases) |alias| {
        if (std.mem.eql(u8, alias.alias, alias_name)) return alias;
    }
    return null;
}

fn modulePathForRootMember(index: *const Index, member_name: []const u8) ?[]const u8 {
    const root = index.module("std") orelse return null;
    const symbol = findSymbol(root.index.symbols, member_name) orelse return null;
    const detail = symbol.detail orelse return null;
    return importPathFromDetail(detail);
}

fn importPathFromDetail(detail: []const u8) ?[]const u8 {
    const prefix = "import \"";
    if (!std.mem.startsWith(u8, detail, prefix)) return null;
    const rest = detail[prefix.len..];
    const end = std.mem.indexOfScalar(u8, rest, '"') orelse return null;
    const logical_path = rest[0..end];
    if (embedded_stdlib.byLogicalPath(logical_path) == null) return null;
    return logical_path;
}

fn findSymbol(symbols: []const semantic_index.Symbol, name: []const u8) ?semantic_index.Symbol {
    for (symbols) |symbol| {
        if (!std.mem.eql(u8, symbol.name, name)) continue;
        if (symbol.kind == .parameter or symbol.kind == .field) continue;
        return symbol;
    }
    return null;
}

fn isAliasToken(token: token_cache.Token) bool {
    if (token.lexeme.len == 0) return false;
    if (!isIdentifierStart(token.lexeme[0])) return false;
    for (token.lexeme[1..]) |ch| {
        if (!isIdentifierContinue(ch)) return false;
    }
    return true;
}

fn accessChainEndingAt(source: []const u8, word: Word) ?Chain {
    var reversed = Chain{};
    if (!reversed.append(word.text)) return null;

    var cursor = word.start_offset;
    while (cursor > 0 and source[cursor - 1] == '.') {
        cursor -= 1;
        if (cursor == 0 or !isIdentifierContinue(source[cursor - 1])) return null;

        const end = cursor;
        var start = end;
        while (start > 0 and isIdentifierContinue(source[start - 1])) : (start -= 1) {}
        if (start == end) return null;
        if (!reversed.append(source[start..end])) return null;
        cursor = start;
    }

    if (reversed.len < 2) return null;

    var chain = Chain{};
    var i = reversed.len;
    while (i > 0) {
        i -= 1;
        if (!chain.append(reversed.at(i))) return null;
    }
    return chain;
}

fn accessChainBeforeDot(source: []const u8, dot_offset: usize) ?Chain {
    if (dot_offset == 0 or source[dot_offset] != '.') return null;

    var reversed = Chain{};
    var cursor = dot_offset;
    while (cursor > 0) {
        if (!isIdentifierContinue(source[cursor - 1])) break;

        const end = cursor;
        var start = end;
        while (start > 0 and isIdentifierContinue(source[start - 1])) : (start -= 1) {}
        if (start == end) return null;
        if (!reversed.append(source[start..end])) return null;

        if (start == 0 or source[start - 1] != '.') break;
        cursor = start - 1;
    }

    if (reversed.len == 0) return null;

    var chain = Chain{};
    var i = reversed.len;
    while (i > 0) {
        i -= 1;
        if (!chain.append(reversed.at(i))) return null;
    }
    return chain;
}

fn wordAtPosition(source: []const u8, position: frontend.Position) ?Word {
    var cursor: usize = 0;
    var line: u32 = 0;
    while (cursor < source.len and line < position.line) : (cursor += 1) {
        if (source[cursor] == '\n') line += 1;
    }
    if (line != position.line) return null;

    const line_start = cursor;
    const col: usize = @intCast(position.character);
    const pos = @min(line_start + col, source.len);

    var start = pos;
    while (start > line_start and isIdentifierContinue(source[start - 1])) start -= 1;
    var end = pos;
    while (end < source.len and source[end] != '\n' and isIdentifierContinue(source[end])) end += 1;
    if (start == end) return null;

    const start_char: u32 = @intCast(start - line_start);
    const end_char: u32 = @intCast(end - line_start);
    return .{
        .text = source[start..end],
        .range = .{
            .start = .{ .line = position.line, .character = start_char },
            .end = .{ .line = position.line, .character = end_char },
        },
        .start_offset = start,
        .end_offset = end,
    };
}

fn identifierPrefixAtOffset(source: []const u8, cursor_offset: usize) []const u8 {
    const cursor = @min(cursor_offset, source.len);
    var start = cursor;
    while (start > 0 and source[start - 1] != '\n' and isIdentifierContinue(source[start - 1])) {
        start -= 1;
    }
    return source[start..cursor];
}

fn positionToByteOffsetOnLine(source: []const u8, position: frontend.Position) usize {
    var cursor: usize = 0;
    var current_line: u32 = 0;

    while (cursor < source.len and current_line < position.line) : (cursor += 1) {
        if (source[cursor] == '\n') current_line += 1;
    }

    if (current_line != position.line) return source.len;

    const line_start = cursor;
    while (cursor < source.len and source[cursor] != '\n') : (cursor += 1) {}
    const line_end = cursor;

    const requested_col: usize = @intCast(position.character);
    return @min(line_start + requested_col, line_end);
}

fn formatHoverValueAlloc(allocator: Allocator, symbol: semantic_index.Symbol) ![]u8 {
    const signature = try formatSignatureAlloc(allocator, symbol);
    defer allocator.free(signature);

    if (symbol.doc_comment) |doc| {
        return std.fmt.allocPrint(allocator, "```ora\n{s}\n```\n---\n{s}", .{ signature, doc });
    }

    return std.fmt.allocPrint(allocator, "```ora\n{s}\n```", .{signature});
}

fn formatSignatureAlloc(allocator: Allocator, symbol: semantic_index.Symbol) ![]u8 {
    return switch (symbol.kind) {
        .function, .method => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "fn {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "fn {s}()", .{symbol.name}),
        .constant => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "const {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "const {s}", .{symbol.name}),
        .variable => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "var {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "var {s}", .{symbol.name}),
        .error_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "error {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "error {s}", .{symbol.name}),
        else => std.fmt.allocPrint(allocator, "{s}", .{symbol.name}),
    };
}

fn isExportedCompletionSymbol(symbol: semantic_index.Symbol) bool {
    return symbol.parent == null and symbol.kind != .parameter and symbol.kind != .field;
}

fn lessSymbolByName(_: void, a: semantic_index.Symbol, b: semantic_index.Symbol) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

fn isIdentifierContinue(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or
        (ch >= 'A' and ch <= 'Z') or
        (ch >= '0' and ch <= '9') or
        ch == '_';
}

fn isIdentifierStart(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or
        (ch >= 'A' and ch <= 'Z') or
        ch == '_';
}
