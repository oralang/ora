const std = @import("std");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");

const Allocator = std.mem.Allocator;

pub const Hover = struct {
    contents: []const u8,
    range: frontend.Range,

    pub fn deinit(self: *Hover, allocator: Allocator) void {
        allocator.free(self.contents);
    }
};

pub fn hoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    var index = try semantic_index.indexDocument(allocator, source);
    defer index.deinit(allocator);

    if (!index.parse_succeeded) return null;

    if (semantic_index.findSymbolAtPosition(index.symbols, position)) |symbol_index| {
        const candidate = index.symbols[symbol_index];
        if (rangeContainsPosition(candidate.selection_range, position)) {
            const value = try formatHoverValueAlloc(allocator, candidate);
            return .{ .contents = value, .range = candidate.selection_range };
        }
    }

    return keywordHoverAt(allocator, source, position);
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
        .contract => std.fmt.allocPrint(allocator, "contract {s}", .{symbol.name}),
        .function, .method => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "fn {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "fn {s}()", .{symbol.name}),
        .variable => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "var {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "var {s}", .{symbol.name}),
        .field => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "field {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "field {s}", .{symbol.name}),
        .constant => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "const {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "const {s}", .{symbol.name}),
        .parameter => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "{s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "{s}", .{symbol.name}),
        .struct_decl => std.fmt.allocPrint(allocator, "struct {s}", .{symbol.name}),
        .bitfield_decl => std.fmt.allocPrint(allocator, "bitfield {s}", .{symbol.name}),
        .enum_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "enum {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "enum {s}", .{symbol.name}),
        .enum_member => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "enum member {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "enum member {s}", .{symbol.name}),
        .trait_decl => std.fmt.allocPrint(allocator, "trait {s}", .{symbol.name}),
        .impl_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "{s} {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "{s}", .{symbol.name}),
        .type_alias => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "type {s} = {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "type {s}", .{symbol.name}),
        .event => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "log {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "log {s}", .{symbol.name}),
        .error_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "error {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "error {s}", .{symbol.name}),
    };
}

fn keywordHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    const doc = keyword_docs.get(word.text) orelse return null;

    const value = try std.fmt.allocPrint(allocator, "```ora\n{s}\n```\n---\n{s}", .{ word.text, doc });
    return .{ .contents = value, .range = word.range };
}

const WordAtPosition = struct { text: []const u8, range: frontend.Range };

fn wordAtPosition(source: []const u8, position: frontend.Position) ?WordAtPosition {
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
    while (start > line_start and isIdentChar(source[start - 1])) start -= 1;
    var end = pos;
    while (end < source.len and source[end] != '\n' and isIdentChar(source[end])) end += 1;
    if (start == end) return null;

    const start_char: u32 = @intCast(start - line_start);
    const end_char: u32 = @intCast(end - line_start);
    return .{
        .text = source[start..end],
        .range = .{
            .start = .{ .line = position.line, .character = start_char },
            .end = .{ .line = position.line, .character = end_char },
        },
    };
}

fn isIdentChar(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z') or (ch >= '0' and ch <= '9') or ch == '_';
}

const keyword_docs = std.StaticStringMap([]const u8).initComptime(.{
    .{ "contract", "Declares a smart contract type." },
    .{ "struct", "Declares a named struct type." },
    .{ "enum", "Declares an enumeration type." },
    .{ "bitfield", "Declares a packed bitfield type for efficient storage." },
    .{ "type", "Declares a type alias." },
    .{ "fn", "Declares a function." },
    .{ "pub", "Makes a declaration publicly visible." },
    .{ "let", "Declares an immutable local binding." },
    .{ "var", "Declares a mutable variable." },
    .{ "const", "Declares a compile-time constant." },
    .{ "storage", "Storage qualifier — persists on-chain between calls." },
    .{ "transient", "Transient storage qualifier — cleared after each transaction." },
    .{ "memory", "Memory qualifier — temporary data within a call." },
    .{ "import", "Imports declarations from another module." },
    .{ "log", "Declares an event (emits an EVM log)." },
    .{ "error", "Declares a custom error type." },
    .{ "trait", "Declares an interface trait." },
    .{ "impl", "Implements a trait for a type." },
    .{ "comptime", "Evaluates an expression at compile time." },
    .{ "if", "Conditional branch." },
    .{ "else", "Alternative branch of an `if` or `switch`." },
    .{ "while", "Loop that repeats while a condition holds." },
    .{ "for", "Iterates over a range or collection." },
    .{ "switch", "Multi-way branch on a value." },
    .{ "match", "Pattern match over values, enums, and Result/error unions." },
    .{ "return", "Returns a value from the current function." },
    .{ "break", "Exits the innermost loop." },
    .{ "continue", "Skips to the next iteration of the innermost loop." },
    .{ "try", "Unwraps an error union, propagating the error on failure." },
    .{ "catch", "Handles an error from an error union." },
    .{ "requires", "Precondition — must hold when the function is called." },
    .{ "guard", "Runtime-enforced precondition — checked at runtime and assumed after it passes." },
    .{ "ensures", "Postcondition — guaranteed to hold when the function returns." },
    .{ "invariant", "Contract or loop invariant — preserved across state transitions." },
    .{ "ghost", "Ghost declaration — exists only for verification, not compiled." },
    .{ "assert", "Verification assertion — checked by the prover." },
    .{ "assume", "Verification assumption — taken as given by the prover." },
    .{ "havoc", "Assigns an arbitrary value for verification." },
    .{ "old", "Refers to the pre-state value of an expression in postconditions." },
    .{ "result", "Refers to the return value in postconditions." },
    .{ "modifies", "Declares state locations a function may modify." },
    .{ "decreases", "Declares a decreasing termination measure." },
    .{ "increases", "Declares an increasing termination measure." },
    .{ "forall", "Universal quantifier — for all values satisfying a predicate." },
    .{ "exists", "Existential quantifier — there exists a value satisfying a predicate." },
    .{ "where", "Type constraint or refinement clause." },
    .{ "extern", "Declares an external contract interface." },
    .{ "true", "Boolean literal `true`." },
    .{ "false", "Boolean literal `false`." },
    .{ "self", "Refers to the current contract instance." },
    .{ "call", "Declares or invokes a state-changing external call." },
    .{ "staticcall", "Declares or invokes a read-only external call." },
    .{ "errors", "Declares the closed error set an extern trait method may return." },
});

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
