const std = @import("std");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");

const Allocator = std.mem.Allocator;

pub const ParameterInfo = struct {
    label: []u8,
};

pub const SignatureInfo = struct {
    label: []u8,
    documentation: ?[]u8 = null,
    parameters: []ParameterInfo,
    active_parameter: u32,

    pub fn deinit(self: *SignatureInfo, allocator: Allocator) void {
        allocator.free(self.label);
        if (self.documentation) |doc| allocator.free(doc);
        for (self.parameters) |param| allocator.free(param.label);
        allocator.free(self.parameters);
    }
};

/// Given source and a cursor position (typically after `(` or `,`), find the
/// function being called and return its signature with the active parameter.
pub fn signatureAt(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
) !?SignatureInfo {
    const byte_offset = positionToByteOffset(source, position);
    if (byte_offset == 0) return null;

    // Scan backwards to find the unmatched `(`.
    const call_ctx = findCallContext(source, byte_offset) orelse return null;

    // Extract the function name before the `(`.
    const callee_name = extractCalleeName(source, call_ctx.paren_offset) orelse return null;

    // Look up the function in the semantic index.
    var index = try semantic_index.indexDocument(allocator, source);
    defer index.deinit(allocator);

    const symbol = findFunctionSymbol(index.symbols, callee_name) orelse return null;

    // Build signature from symbol detail.
    return try buildSignatureInfo(allocator, symbol, call_ctx.active_parameter);
}

const CallContext = struct {
    paren_offset: usize,
    active_parameter: u32,
};

fn findCallContext(source: []const u8, cursor: usize) ?CallContext {
    var depth: i32 = 0;
    var comma_count: u32 = 0;
    var i: usize = cursor;

    // Start scanning backwards from cursor.
    while (i > 0) {
        i -= 1;
        const ch = source[i];

        switch (ch) {
            ')' => depth += 1,
            ']' => depth += 1,
            '}' => depth += 1,
            '(' => {
                if (depth == 0) {
                    return .{
                        .paren_offset = i,
                        .active_parameter = comma_count,
                    };
                }
                depth -= 1;
            },
            '[' => {
                if (depth > 0) depth -= 1;
            },
            '{' => {
                if (depth > 0) depth -= 1;
            },
            ',' => {
                if (depth == 0) comma_count += 1;
            },
            else => {},
        }
    }

    return null;
}

fn extractCalleeName(source: []const u8, paren_offset: usize) ?[]const u8 {
    if (paren_offset == 0) return null;

    // Skip whitespace before `(`.
    var end = paren_offset;
    while (end > 0 and (source[end - 1] == ' ' or source[end - 1] == '\t')) {
        end -= 1;
    }
    if (end == 0) return null;

    // Walk backwards through identifier characters.
    var start = end;
    while (start > 0 and isIdentifierContinue(source[start - 1])) {
        start -= 1;
    }

    if (start == end) return null;
    return source[start..end];
}

fn findFunctionSymbol(symbols: []const semantic_index.Symbol, name: []const u8) ?*const semantic_index.Symbol {
    // Prefer function/method over other kinds.
    var best: ?*const semantic_index.Symbol = null;
    for (symbols) |*symbol| {
        if (!std.mem.eql(u8, symbol.name, name)) continue;
        if (symbol.kind == .function or symbol.kind == .method) return symbol;
        if (symbol.kind == .event or symbol.kind == .error_decl) {
            if (best == null) best = symbol;
        }
    }
    return best;
}

fn buildSignatureInfo(allocator: Allocator, symbol: *const semantic_index.Symbol, active_param: u32) !SignatureInfo {
    const detail = symbol.detail orelse return error.OutOfMemory;

    // Parse parameter labels from the detail string, e.g. "(x: u256, y: address) -> bool"
    var params = std.ArrayList(ParameterInfo){};
    errdefer {
        for (params.items) |p| allocator.free(p.label);
        params.deinit(allocator);
    }

    if (detail.len > 0 and detail[0] == '(') {
        var i: usize = 1;
        while (i < detail.len and detail[i] != ')') {
            const param_start = i;
            var depth: u32 = 0;
            while (i < detail.len) : (i += 1) {
                if (detail[i] == '(' or detail[i] == '<') {
                    depth += 1;
                } else if (detail[i] == ')' or detail[i] == '>') {
                    if (depth == 0) break;
                    depth -= 1;
                } else if (detail[i] == ',' and depth == 0) {
                    break;
                }
            }
            const param_text = std.mem.trim(u8, detail[param_start..i], " ");
            if (param_text.len > 0) {
                try params.append(allocator, .{
                    .label = try allocator.dupe(u8, param_text),
                });
            }
            if (i < detail.len and detail[i] == ',') i += 1;
            // Skip whitespace after comma.
            while (i < detail.len and detail[i] == ' ') : (i += 1) {}
        }
    }

    // Build the full signature label.
    const kind_prefix: []const u8 = switch (symbol.kind) {
        .function, .method => "fn ",
        .event => "log ",
        .error_decl => "error ",
        else => "",
    };
    const label = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ kind_prefix, symbol.name, detail });
    errdefer allocator.free(label);

    const doc: ?[]u8 = if (symbol.doc_comment) |dc| try allocator.dupe(u8, dc) else null;

    return .{
        .label = label,
        .documentation = doc,
        .parameters = try params.toOwnedSlice(allocator),
        .active_parameter = active_param,
    };
}

fn positionToByteOffset(source: []const u8, position: frontend.Position) usize {
    var cursor: usize = 0;
    var current_line: u32 = 0;

    while (cursor < source.len and current_line < position.line) : (cursor += 1) {
        if (source[cursor] == '\n') current_line += 1;
    }
    if (current_line != position.line) return source.len;

    const col: usize = @intCast(position.character);
    return @min(cursor + col, source.len);
}

fn isIdentifierContinue(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z') or (ch >= '0' and ch <= '9') or ch == '_';
}
