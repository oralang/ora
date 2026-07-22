const std = @import("std");
const testing = std.testing;

const formatting = @import("formatting.zig");

test "lsp formatting: formats and is idempotent" {
    const source =
        \\contract Wallet{
        \\pub fn run()->u256{return 1;}
        \\}
    ;

    const formatted_once = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_once);

    try testing.expect(!std.mem.eql(u8, source, formatted_once));

    const formatted_twice = try formatting.formatSourceAlloc(testing.allocator, formatted_once, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_twice);

    try testing.expectEqualStrings(formatted_once, formatted_twice);
}

test "lsp formatting: preserves comments around directive statements" {
    // Regression: the lexer attached a scan step trivia to its LAST token,
    // so a comment before `@callHint` hung on the name token, where the
    // formatter post-`@` spacing rule silently deleted it on save.
    const source =
        \\contract C {
        \\    storage var x: u256;
        \\
        \\    pub fn transferFrom(v: u256) {
        \\        // transferFrom is a hot path for ERC20 usage
        \\        @callHint(likely);
        \\        x = v;
        \\    }
        \\}
    ;

    const formatted = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted);

    try testing.expect(std.mem.indexOf(u8, formatted, "// transferFrom is a hot path for ERC20 usage") != null);
    // The comment stays on its own line, above the directive.
    try testing.expect(std.mem.indexOf(u8, formatted, "usage\n        @callHint(likely);") != null);
}

test "lsp formatting: keeps same-line trailing comments on their line" {
    // Regression: formatSemicolon wrote its newline eagerly, pushing
    // `; // why` comments onto the next line.
    const source =
        \\contract C {
        \\    storage var x: u256;
        \\
        \\    pub fn mint(v: u256) {
        \\        @callHint(cold); // mint is not a hot-path priority
        \\        x = v; // trailing on assign
        \\    }
        \\}
    ;

    const formatted = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted);

    try testing.expect(std.mem.indexOf(u8, formatted, "@callHint(cold); // mint is not a hot-path priority") != null);
    try testing.expect(std.mem.indexOf(u8, formatted, "x = v; // trailing on assign") != null);

    const twice = try formatting.formatSourceAlloc(testing.allocator, formatted, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(twice);
    try testing.expectEqualStrings(formatted, twice);
}

test "lsp formatting: parse errors surface as ParseError" {
    const invalid = "@import(\"std\");";
    try testing.expectError(
        error.ParseError,
        formatting.formatSourceAlloc(testing.allocator, invalid, .{}),
    );
}

test "lsp formatting: preserves type alias spacing" {
    const source =
        \\type Amount = u256;
        \\type AmountArray4 = [Amount; 4];
        \\type AliasArray(comptime T: type, comptime N: u256) = [T; N];
    ;

    const formatted = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted);

    try testing.expectEqualStrings(source ++ "\n", formatted);
}

test "lsp formatting: preserves keyword boundaries for inline impl and modifies" {
    const source =
        \\trait SettledBalance {
        \\    fn settled(self: Settled) -> u256;
        \\}
        \\
        \\struct Settled {
        \\    value: u256;
        \\}
        \\
        \\impl SettledBalance for Settled {
        \\    fn settled(self: Settled) -> u256 {
        \\        return self.value;
        \\    }
        \\}
        \\
        \\contract Token {
        \\    inline fn amountDelta(amount: u256) -> u256 {
        \\        return amount;
        \\    }
        \\
        \\    pub fn approve(spender: address, amount: u256)
        \\        modifies allowances[std.msg.sender()][spender]
        \\    {
        \\        return;
        \\    }
        \\}
    ;

    const formatted_once = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_once);

    const formatted_twice = try formatting.formatSourceAlloc(testing.allocator, formatted_once, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_twice);

    try testing.expectEqualStrings(formatted_once, formatted_twice);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "inline fn amountDelta") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "impl SettledBalance for Settled") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "modifies allowances[std.msg.sender()][spender]") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_twice, "forfor") == null);
    try testing.expect(std.mem.indexOf(u8, formatted_twice, "inlinefn") == null);
    try testing.expect(std.mem.indexOf(u8, formatted_twice, "modifiesallowances") == null);

    const corrupted =
        \\contract Token {
        \\    pub fn approve(spender: address, amount: u256)
        \\    modifiesallowances[std.msg.sender()][spender]
        \\    {
        \\        return;
        \\    }
        \\}
    ;

    const repaired = try formatting.formatSourceAlloc(testing.allocator, corrupted, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(repaired);

    try testing.expect(std.mem.indexOf(u8, repaired, "modifies allowances[std.msg.sender()][spender]") != null);
    try testing.expect(std.mem.indexOf(u8, repaired, "modifiesallowances") == null);
}

test "lsp formatting: preserves resource error-union and builtin syntax" {
    const source =
        \\comptime const std = @import("std");
        \\
        \\resource TokenUnit = u256;
        \\
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract ERC20Token {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    inline fn requireBalance(owner: address, amount: TokenUnit) -> !bool | InsufficientBalance {
        \\        let available: TokenUnit = @amount(balances[owner]);
        \\        if (available < amount) {
        \\            return InsufficientBalance(amount, available);
        \\        }
        \\        return true;
        \\    }
        \\
        \\    pub fn transfer(recipient: NonZeroAddress, amount: TokenUnit) -> !bool | InsufficientBalance {
        \\        try requireBalance(std.msg.sender(), amount);
        \\        @move(balances[std.msg.sender()], balances[recipient], amount);
        \\        return true;
        \\    }
        \\}
    ;

    const formatted_once = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_once);

    const formatted_twice = try formatting.formatSourceAlloc(testing.allocator, formatted_once, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_twice);

    try testing.expectEqualStrings(formatted_once, formatted_twice);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "resource TokenUnit = u256;") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "@amount(balances[owner])") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "-> !bool | InsufficientBalance") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "@move(balances[std.msg.sender()], balances[recipient], amount);") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "resourceTokenUnit") == null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "!bool|") == null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "@ move") == null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "@\n") == null);
}

const ora_lib = @import("ora_lib");

fn formatAndExpect(source: []const u8, expected_needle: []const u8) !void {
    const formatted = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted);
    try testing.expect(std.mem.containsAtLeast(u8, formatted, 1, expected_needle));

    const twice = try formatting.formatSourceAlloc(testing.allocator, formatted, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(twice);
    try testing.expectEqualStrings(formatted, twice);
}

const BinaryOpSpacing = struct {
    op: ora_lib.compiler.ast.nodes.BinaryOp,
    glued: []const u8,
    spaced: []const u8,
};

const binary_op_spacing_table = [_]BinaryOpSpacing{
    .{ .op = .add, .glued = "a+b", .spaced = "a + b" },
    .{ .op = .wrapping_add, .glued = "a+%b", .spaced = "a +% b" },
    .{ .op = .sub, .glued = "a-b", .spaced = "a - b" },
    .{ .op = .wrapping_sub, .glued = "a-%b", .spaced = "a -% b" },
    .{ .op = .mul, .glued = "a*b", .spaced = "a * b" },
    .{ .op = .wrapping_mul, .glued = "a*%b", .spaced = "a *% b" },
    .{ .op = .div, .glued = "a/b", .spaced = "a / b" },
    .{ .op = .mod, .glued = "a%b", .spaced = "a % b" },
    .{ .op = .pow, .glued = "a**b", .spaced = "a ** b" },
    .{ .op = .wrapping_pow, .glued = "a**%b", .spaced = "a **% b" },
    .{ .op = .eq, .glued = "a==b", .spaced = "a == b" },
    .{ .op = .ne, .glued = "a!=b", .spaced = "a != b" },
    .{ .op = .lt, .glued = "a<b", .spaced = "a < b" },
    .{ .op = .le, .glued = "a<=b", .spaced = "a <= b" },
    .{ .op = .gt, .glued = "a>b", .spaced = "a > b" },
    .{ .op = .ge, .glued = "a>=b", .spaced = "a >= b" },
    .{ .op = .and_and, .glued = "c&&d", .spaced = "c && d" },
    .{ .op = .or_or, .glued = "c||d", .spaced = "c || d" },
    .{ .op = .bit_and, .glued = "a&b", .spaced = "a & b" },
    .{ .op = .bit_or, .glued = "a|b", .spaced = "a | b" },
    .{ .op = .bit_xor, .glued = "a^b", .spaced = "a ^ b" },
    .{ .op = .shl, .glued = "a<<b", .spaced = "a << b" },
    .{ .op = .wrapping_shl, .glued = "a<<%b", .spaced = "a <<% b" },
    .{ .op = .shr, .glued = "a>>b", .spaced = "a >> b" },
    .{ .op = .wrapping_shr, .glued = "a>>%b", .spaced = "a >>% b" },
};

comptime {
    const fields = @typeInfo(ora_lib.compiler.ast.nodes.BinaryOp).@"enum".fields;
    if (fields.len != binary_op_spacing_table.len)
        @compileError("binary_op_spacing_table must cover every ast.BinaryOp variant");
    for (fields) |field| {
        var found = false;
        for (binary_op_spacing_table) |row| {
            if (@intFromEnum(row.op) == field.value) found = true;
        }
        if (!found) @compileError("binary_op_spacing_table missing op: " ++ field.name);
    }
}

test "lsp formatting: every binary operator is spaced" {
    inline for (binary_op_spacing_table) |row| {
        const source = try std.fmt.allocPrint(testing.allocator,
            \\contract T {{
            \\    pub fn f(a: u256, b: u256, c: bool, d: bool) -> u256 {{
            \\        let r = {s};
            \\        return a;
            \\    }}
            \\}}
        , .{row.glued});
        defer testing.allocator.free(source);
        try formatAndExpect(source, row.spaced);
    }
}

test "lsp formatting: generic angle brackets stay tight" {
    const source =
        \\contract T {
        \\    storage var one: map<address, u256>;
        \\    storage var two: map<u256, map<u256, u256>>;
        \\    pub fn f(a: u256, b: u256) -> u256 {
        \\        let cmp = a<b;
        \\        let shifted = a>>b;
        \\        return a;
        \\    }
        \\}
    ;
    const formatted = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted);
    try testing.expect(std.mem.containsAtLeast(u8, formatted, 1, "map<address, u256>"));
    try testing.expect(std.mem.containsAtLeast(u8, formatted, 1, "map<u256, map<u256, u256>>"));
    try testing.expect(std.mem.containsAtLeast(u8, formatted, 1, "a < b"));
    try testing.expect(std.mem.containsAtLeast(u8, formatted, 1, "a >> b"));
}

test "lsp formatting: reported operator regressions" {
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(executed: u256, cycles: u256) -> bool {
        \\        return executed<cycles;
        \\    }
        \\}
    , "executed < cycles");
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(op: u256) -> u256 {
        \\        switch (op>>12) {
        \\            0 => { return 1; },
        \\            else => { return 0; }
        \\        }
        \\    }
        \\}
    , "switch (op >> 12)");
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(sp: u256) -> u256 {
        \\        var call_stack: [u256; 4] = [0, 0, 0, 0];
        \\        return call_stack[sp]&0xFFF;
        \\    }
        \\}
    , "call_stack[sp] & 0xFFF");
}

test "lsp formatting: prefix operators keep operands tight" {
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(a: u256) -> u256 {
        \\        let n = -1;
        \\        return a;
        \\    }
        \\}
    , "let n = -1;");
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(c: bool) -> bool {
        \\        return !c;
        \\    }
        \\}
    , "return !c;");
}

test "lsp formatting: comparison spacing in spec clauses" {
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(key: u256) -> u256
        \\        requires(key<16)
        \\    {
        \\        var i: u256 = 0;
        \\        while (i<key)
        \\            invariant(i<=key)
        \\        {
        \\            i = i+1;
        \\        }
        \\        return i;
        \\    }
        \\}
    , "requires(key < 16)");
}

test "lsp formatting: operator followed by parenthesis or bracket is spaced" {
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(a: u256, b: u256) -> u256 {
        \\        let p =(a<b);
        \\        return a;
        \\    }
        \\}
    , "let p = (a < b);");
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(a: u256, b: u256) -> u256 {
        \\        return a+(b*2);
        \\    }
        \\}
    , "return a + (b * 2);");
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(keys: u256, x: u256) -> u256 {
        \\        var v: [u256; 4] = [0, 0, 0, 0];
        \\        return keys>>(v[x]&0xF);
        \\    }
        \\}
    , "return keys >> (v[x] & 0xF);");
    try formatAndExpect(
        \\contract T {
        \\    pub fn f(a: u256) -> u256 {
        \\        let arr =[1, 2];
        \\        return a;
        \\    }
        \\}
    , "let arr = [1, 2];");
}

test "lsp formatting: call parentheses stay tight" {
    try formatAndExpect(
        \\contract T {
        \\    fn g(a: u256) -> u256 { return a; }
        \\    pub fn f(a: u256) -> u256 {
        \\        return g(a);
        \\    }
        \\}
    , "return g(a);");
}
