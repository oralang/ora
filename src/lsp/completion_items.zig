const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const completion = ora_root.lsp.completion;
const frontend = ora_root.lsp.frontend;

const Allocator = std.mem.Allocator;
const types = lsp.types;

const Snippet = struct {
    label: []const u8,
    body: []const u8,
    detail: []const u8,
};

const snippets = [_]Snippet{
    .{ .label = "contract", .body = "contract ${1:Name} {\n\t$0\n}", .detail = "Contract declaration" },
    .{ .label = "fn", .body = "fn ${1:name}(${2}) -> ${3:u256} {\n\t$0\n}", .detail = "Function declaration" },
    .{ .label = "pub fn", .body = "pub fn ${1:name}(${2}) -> ${3:u256} {\n\t$0\n}", .detail = "Public function declaration" },
    .{ .label = "if", .body = "if (${1:condition}) {\n\t$0\n}", .detail = "If statement" },
    .{ .label = "while", .body = "while (${1:condition}) {\n\t$0\n}", .detail = "While loop" },
    .{ .label = "for", .body = "for (${1:item} in ${2:iterable}) {\n\t$0\n}", .detail = "For loop" },
    .{ .label = "struct", .body = "struct ${1:Name} {\n\t${2:field}: ${3:u256},\n}", .detail = "Struct declaration" },
    .{ .label = "requires", .body = "requires(${1:condition})", .detail = "Precondition clause" },
    .{ .label = "ensures", .body = "ensures(${1:condition})", .detail = "Postcondition clause" },
    .{ .label = "import", .body = "const ${1:name} = @import(\"${2:path}.ora\");", .detail = "Import declaration" },
    .{ .label = "storage", .body = "storage var ${1:name}: ${2:u256};", .detail = "Storage variable" },
    .{ .label = "event", .body = "log ${1:Name}(${2:param}: ${3:u256});", .detail = "Event/log declaration" },
};

pub fn build(
    arena: Allocator,
    source: []const u8,
    byte_position: frontend.Position,
    trigger_char: ?[]const u8,
    items: []const completion.Item,
) ![]types.CompletionItem {
    const snippet_count: usize = if (isAtLineStart(source, byte_position) and trigger_char == null) snippets.len else 0;
    const result = try arena.alloc(types.CompletionItem, items.len + snippet_count);

    for (items, 0..) |item, i| {
        result[i] = .{
            .label = item.label,
            .kind = kindToLsp(item.kind),
            .detail = item.detail,
            .documentation = if (item.documentation) |doc| .{ .MarkupContent = .{
                .kind = .markdown,
                .value = doc,
            } } else null,
        };
    }

    for (snippets[0..snippet_count], items.len..) |snippet, i| {
        result[i] = .{
            .label = snippet.label,
            .kind = .Snippet,
            .detail = snippet.detail,
            .insertTextFormat = .Snippet,
            .insertText = snippet.body,
        };
    }

    return result;
}

fn isAtLineStart(source: []const u8, position: frontend.Position) bool {
    var line: u32 = 0;
    var line_start: usize = 0;
    for (source, 0..) |c, i| {
        if (line == position.line) {
            const prefix = source[line_start..@min(line_start + position.character, source.len)];
            for (prefix) |ch| {
                if (ch != ' ' and ch != '\t') return false;
            }
            return true;
        }
        if (c == '\n') {
            line += 1;
            line_start = i + 1;
        }
    }
    return line == position.line;
}

fn kindToLsp(kind: completion.Kind) types.CompletionItemKind {
    return switch (kind) {
        .keyword => .Keyword,
        .contract => .Class,
        .function => .Function,
        .method => .Method,
        .variable => .Variable,
        .field => .Field,
        .constant => .Constant,
        .parameter => .Variable,
        .struct_decl => .Struct,
        .bitfield_decl => .Struct,
        .enum_decl => .Enum,
        .enum_member => .EnumMember,
        .trait_decl => .Interface,
        .impl_decl => .Class,
        .type_alias => .Struct,
        .event => .Event,
        .error_decl => .Class,
    };
}
