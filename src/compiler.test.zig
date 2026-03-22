const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;
const mlir = @import("mlir_c_api").c;

fn compileText(source_text: []const u8) !compiler.driver.Compilation {
    return compiler.compileSource(testing.allocator, "test.ora", source_text);
}

fn renderHirTextForSource(source_text: []const u8) ![]u8 {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    return hir_result.renderText(testing.allocator);
}

fn compilePackage(root_path: []const u8) !compiler.driver.Compilation {
    return compiler.compilePackage(testing.allocator, root_path);
}

fn expectOraToSirConverts(path: []const u8) !void {
    var compilation = try compilePackage(path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
}

fn expectNoResidualOraRuntimeOps(rendered: []const u8) !void {
    const forbidden = [_][]const u8{
        "ora.global",
        "ora.sload",
        "ora.sstore",
        "ora.tload",
        "ora.tstore",
        "ora.map_get",
        "ora.map_store",
        "ora.return",
        "ora.error.ok",
        "ora.error.err",
        "ora.error.is_error",
        "ora.error.unwrap",
        "ora.error.get_error",
        "ora.error.return",
        "ora.if",
        "ora.try_stmt",
        "ora.switch",
        "ora.yield",
        "ora.break",
        "ora.continue",
        "ora.conditional_return",
        "ora.struct_instantiate",
        "ora.struct_field_extract",
        "ora.struct_field_update",
        "ora.struct.decl",
        "ora.tuple_create",
        "ora.tuple_extract",
        "ora.abi_encode",
        "ora.external_call",
        "ora.abi_decode",
        "ora.assert",
        "ora.log",
        "ora.lock",
        "ora.unlock",
        "ora.refinement_to_base",
        "ora.base_to_refinement",
    };

    for (forbidden) |needle| {
        const as_result = try std.fmt.allocPrint(testing.allocator, "= {s}", .{needle});
        defer testing.allocator.free(as_result);
        const as_stmt = try std.fmt.allocPrint(testing.allocator, "\n    {s}", .{needle});
        defer testing.allocator.free(as_stmt);
        if (std.mem.containsAtLeast(u8, rendered, 1, as_result) or std.mem.containsAtLeast(u8, rendered, 1, as_stmt)) {
            return error.TestUnexpectedResult;
        }
    }
}

fn firstChildNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind) ?compiler.SyntaxNode {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| if (child_node.kind() == kind) return child_node,
            .token => {},
        }
    }
    return null;
}

fn nthChildNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, ordinal: usize) ?compiler.SyntaxNode {
    var seen: usize = 0;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| {
                if (child_node.kind() != kind) continue;
                if (seen == ordinal) return child_node;
                seen += 1;
            },
            .token => {},
        }
    }
    return null;
}

fn containsNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind) bool {
    if (node.kind() == kind) return true;

    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| if (containsNodeOfKind(child_node, kind)) return true,
            .token => {},
        }
    }
    return false;
}

fn findVariablePatternByName(ast_file: *const compiler.ast.AstFile, statements: []const compiler.ast.StmtId, name: []const u8) ?compiler.ast.PatternId {
    for (statements) |statement_id| {
        const statement = ast_file.statement(statement_id).*;
        if (statement != .VariableDecl) continue;
        const pattern_id = statement.VariableDecl.pattern;
        const pattern = ast_file.pattern(pattern_id).*;
        if (pattern != .Name) continue;
        if (std.mem.eql(u8, pattern.Name.name, name)) return pattern_id;
    }
    return null;
}

fn diagnosticMessagesContain(diags: *const compiler.diagnostics.DiagnosticList, needle: []const u8) bool {
    for (diags.items.items) |diag| {
        if (std.mem.containsAtLeast(u8, diag.message, 1, needle)) return true;
    }
    return false;
}

fn countDiagnosticMessages(diags: *const compiler.diagnostics.DiagnosticList, needle: []const u8) usize {
    var count: usize = 0;
    for (diags.items.items) |diag| {
        if (std.mem.eql(u8, diag.message, needle)) count += 1;
    }
    return count;
}

fn containsEffectSlot(items: []const compiler.sema.EffectSlot, needle: []const u8, region: compiler.sema.Region) bool {
    for (items) |item| {
        if (item.region == region and std.mem.eql(u8, item.name, needle)) return true;
    }
    return false;
}

fn containsKeyedEffectSlot(items: []const compiler.sema.EffectSlot, needle: []const u8, region: compiler.sema.Region, key_path: []const compiler.sema.KeySegment) bool {
    for (items) |item| {
        if (item.region != region) continue;
        if (!std.mem.eql(u8, item.name, needle)) continue;
        const item_path = item.key_path orelse continue;
        if (item_path.len != key_path.len) continue;
        var all_match = true;
        for (item_path, key_path) |lhs, rhs| {
            if (!std.meta.eql(lhs, rhs)) {
                all_match = false;
                break;
            }
        }
        if (all_match) return true;
    }
    return false;
}

fn nthDescendantNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, ordinal: usize) ?compiler.SyntaxNode {
    var remaining = ordinal;
    return nthDescendantNodeOfKindInner(node, kind, &remaining);
}

fn nthDescendantNodeOfKindInner(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, remaining: *usize) ?compiler.SyntaxNode {
    if (node.kind() == kind) {
        if (remaining.* == 0) return node;
        remaining.* -= 1;
    }

    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| {
                if (nthDescendantNodeOfKindInner(child_node, kind, remaining)) |found| return found;
            },
            .token => {},
        }
    }
    return null;
}

test "compiler syntax preserves source and syntax pointers resolve" {
    const source_text =
        \\// leading comment
        \\contract Test {
        \\    pub fn run() -> u256 {
        \\        return 42;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const rebuilt = try tree.reconstructSource(testing.allocator);
    defer testing.allocator.free(rebuilt);

    try testing.expectEqualStrings(source_text, rebuilt);

    const root = compiler.syntax.rootNode(tree);
    const ptr = root.ptr();
    try testing.expect(ptr.resolve(tree) != null);

    const contract = firstChildNodeOfKind(root, .ContractItem);
    try testing.expect(contract != null);

    const function = firstChildNodeOfKind(contract.?, .FunctionItem);
    try testing.expect(function != null);

    const body = firstChildNodeOfKind(function.?, .Body);
    try testing.expect(body != null);
}

test "compiler syntax parses statement-level bodies" {
    const source_text =
        \\pub fn run(values: u256) -> u256 {
        \\    if (values > 0) {
        \\        return values;
        \\    } else {
        \\        while (values > 1) {
        \\            break;
        \\        }
        \\    }
        \\
        \\    switch (values) {
        \\        0 => return 0;
        \\        else => {
        \\            continue;
        \\        }
        \\    }
        \\
        \\    try {
        \\        havoc values;
        \\    } catch (err) {
        \\        assert(values >= 0, "non-negative");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = firstChildNodeOfKind(root, .FunctionItem);
    try testing.expect(function != null);

    const body = firstChildNodeOfKind(function.?, .Body);
    try testing.expect(body != null);
    try testing.expect(nthChildNodeOfKind(body.?, .IfStmt, 0) != null);
    try testing.expect(nthChildNodeOfKind(body.?, .SwitchStmt, 0) != null);
    try testing.expect(nthChildNodeOfKind(body.?, .TryStmt, 0) != null);

    const if_stmt = nthChildNodeOfKind(body.?, .IfStmt, 0).?;
    const then_body = nthChildNodeOfKind(if_stmt, .Body, 0);
    const else_body = nthChildNodeOfKind(if_stmt, .Body, 1);
    try testing.expect(then_body != null);
    try testing.expect(else_body != null);
    try testing.expect(firstChildNodeOfKind(then_body.?, .ReturnStmt) != null);
    try testing.expect(firstChildNodeOfKind(else_body.?, .WhileStmt) != null);

    const switch_stmt = nthChildNodeOfKind(body.?, .SwitchStmt, 0).?;
    const first_arm = nthChildNodeOfKind(switch_stmt, .SwitchArm, 0);
    const second_arm = nthChildNodeOfKind(switch_stmt, .SwitchArm, 1);
    try testing.expect(first_arm != null);
    try testing.expect(second_arm != null);
    try testing.expect(containsNodeOfKind(first_arm.?, .ExprStmt));

    const try_stmt = nthChildNodeOfKind(body.?, .TryStmt, 0).?;
    const try_body = nthChildNodeOfKind(try_stmt, .Body, 0);
    const catch_clause = nthChildNodeOfKind(try_stmt, .CatchClause, 0);
    const catch_body = if (catch_clause) |clause| nthChildNodeOfKind(clause, .Body, 0) else null;
    try testing.expect(try_body != null);
    try testing.expect(catch_clause != null);
    try testing.expect(catch_body != null);
    try testing.expect(firstChildNodeOfKind(try_body.?, .HavocStmt) != null);
    try testing.expect(firstChildNodeOfKind(catch_body.?, .AssertStmt) != null);
}

test "compiler syntax bounds spec clauses loop invariants and item members" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256;
        \\}
        \\
        \\bitfield Flags: u256 {
        \\    enabled: bool @bits(0..1);
        \\}
        \\
        \\enum Mode {
        \\    Off,
        \\    On,
        \\}
        \\
        \\pub fn run(values: u256) -> u256
        \\    requires values >= 0
        \\    ensures result >= 0
        \\{
        \\    while (values > 0)
        \\        invariant values >= 0;
        \\    {
        \\        break;
        \\    }
        \\
        \\    for (values) |value|
        \\        invariant value >= 0;
        \\    {
        \\        continue;
        \\    }
        \\
        \\    return values;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    const struct_item = nthChildNodeOfKind(root, .StructItem, 0);
    const bitfield_item = nthChildNodeOfKind(root, .BitfieldItem, 0);
    const enum_item = nthChildNodeOfKind(root, .EnumItem, 0);
    const function = nthChildNodeOfKind(root, .FunctionItem, 0);
    try testing.expect(struct_item != null);
    try testing.expect(bitfield_item != null);
    try testing.expect(enum_item != null);
    try testing.expect(function != null);

    try testing.expect(nthChildNodeOfKind(struct_item.?, .StructField, 0) != null);
    try testing.expect(nthChildNodeOfKind(struct_item.?, .StructField, 1) != null);
    try testing.expect(nthChildNodeOfKind(bitfield_item.?, .BitfieldField, 0) != null);
    try testing.expect(nthChildNodeOfKind(enum_item.?, .EnumVariant, 0) != null);
    try testing.expect(nthChildNodeOfKind(enum_item.?, .EnumVariant, 1) != null);

    const requires_clause = nthChildNodeOfKind(function.?, .SpecClause, 0);
    const ensures_clause = nthChildNodeOfKind(function.?, .SpecClause, 1);
    const body = nthChildNodeOfKind(function.?, .Body, 0);
    try testing.expect(requires_clause != null);
    try testing.expect(ensures_clause != null);
    try testing.expect(body != null);
    try testing.expect(firstChildNodeOfKind(requires_clause.?, .Body) == null);
    try testing.expect(firstChildNodeOfKind(ensures_clause.?, .Body) == null);

    const while_stmt = nthChildNodeOfKind(body.?, .WhileStmt, 0);
    const for_stmt = nthChildNodeOfKind(body.?, .ForStmt, 0);
    try testing.expect(while_stmt != null);
    try testing.expect(for_stmt != null);
    try testing.expect(nthChildNodeOfKind(while_stmt.?, .InvariantClause, 0) != null);
    try testing.expect(nthChildNodeOfKind(for_stmt.?, .InvariantClause, 0) != null);
}

test "compiler syntax parses trait and impl blocks" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\    fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    const trait_item = nthChildNodeOfKind(root, .TraitItem, 0);
    const impl_item = nthChildNodeOfKind(root, .ImplItem, 0);
    try testing.expect(trait_item != null);
    try testing.expect(impl_item != null);
    try testing.expect(nthChildNodeOfKind(trait_item.?, .TraitMethodSignature, 0) != null);
    try testing.expect(nthChildNodeOfKind(trait_item.?, .TraitMethodSignature, 1) != null);
    try testing.expect(nthChildNodeOfKind(impl_item.?, .FunctionItem, 0) != null);
}

test "compiler lowers trait and impl items into AST" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\    comptime fn decimals() -> u8;
        \\}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);

    try testing.expect(ast_file.item(ast_file.root_items[0]).* == .Trait);
    const trait_item = ast_file.item(ast_file.root_items[0]).Trait;
    try testing.expectEqualStrings("ERC20", trait_item.name);
    try testing.expect(!trait_item.is_extern);
    try testing.expectEqual(@as(usize, 2), trait_item.methods.len);
    try testing.expectEqual(@as(?compiler.ast.ItemId, null), trait_item.ghost_block);
    try testing.expect(trait_item.methods[0].has_self);
    try testing.expectEqualStrings("totalSupply", trait_item.methods[0].name);
    try testing.expectEqual(@as(usize, 0), trait_item.methods[0].parameters.len);
    try testing.expectEqual(compiler.ast.ExternCallKind.none, trait_item.methods[0].extern_call_kind);
    try testing.expect(trait_item.methods[1].is_comptime);
    try testing.expectEqualStrings("decimals", trait_item.methods[1].name);

    try testing.expect(ast_file.item(ast_file.root_items[1]).* == .Impl);
    const impl_item = ast_file.item(ast_file.root_items[1]).Impl;
    try testing.expectEqualStrings("ERC20", impl_item.trait_name);
    try testing.expectEqualStrings("Token", impl_item.target_name);
    try testing.expectEqual(@as(usize, 1), impl_item.methods.len);
    try testing.expect(ast_file.item(impl_item.methods[0]).* == .Function);
}

test "compiler parses and lowers extern traits" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(InsufficientBalance, InvalidRecipient);
        \\    staticcall fn totalSupply(self) -> u256;
        \\}
    ;

    var parser_result = try compiler.syntax.parse(testing.allocator, compiler.FileId.fromIndex(0), source_text);
    defer parser_result.deinit();

    const root = compiler.syntax.rootNode(&parser_result.tree);
    const trait_item_node = nthChildNodeOfKind(root, .TraitItem, 0).?;
    try testing.expect(nthChildNodeOfKind(trait_item_node, .TraitMethodSignature, 0) != null);

    var ast_diags: std.ArrayList(compiler.diagnostics.Diagnostic) = .{};
    defer ast_diags.deinit(testing.allocator);
    var lower_result = try compiler.ast.lower(testing.allocator, &parser_result.tree);
    defer lower_result.deinit();
    try ast_diags.appendSlice(testing.allocator, lower_result.diagnostics.items.items);
    const ast_file = &lower_result.file;
    try testing.expectEqual(@as(usize, 0), ast_diags.items.len);

    const trait_item = ast_file.item(ast_file.root_items[0]).Trait;
    try testing.expect(trait_item.is_extern);
    try testing.expectEqual(@as(usize, 2), trait_item.methods.len);
    try testing.expectEqual(compiler.ast.ExternCallKind.call, trait_item.methods[0].extern_call_kind);
    try testing.expectEqual(compiler.ast.ExternCallKind.staticcall, trait_item.methods[1].extern_call_kind);
    try testing.expectEqual(@as(usize, 2), trait_item.methods[0].errors.len);
    try testing.expectEqualStrings("InsufficientBalance", trait_item.methods[0].errors[0]);
    try testing.expectEqualStrings("InvalidRecipient", trait_item.methods[0].errors[1]);
    try testing.expectEqual(@as(usize, 0), trait_item.methods[1].errors.len);
}

test "compiler rejects invalid extern trait semantics" {
    const source_text =
        \\extern trait Bad {
        \\    fn missing(self) -> bool;
        \\    ghost {
        \\        assert(true, "nope");
        \\    }
        \\}
        \\
        \\struct Box { value: u256 }
        \\
        \\impl Bad for Box {
        \\    fn missing(self) -> bool { return true; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &typecheck.diagnostics;
    try testing.expect(diagnosticMessagesContain(diags, "extern trait method 'missing' must use 'call fn' or 'staticcall fn'"));
    try testing.expect(diagnosticMessagesContain(diags, "extern trait 'Bad' cannot declare a ghost block"));
    try testing.expect(diagnosticMessagesContain(diags, "extern trait 'Bad' cannot be implemented with an impl block"));
}

test "compiler type checks external proxy method calls" {
    const source_text =
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn probe(user: address) {
        \\        let call_result = external<ERC20>(token, gas: 50000).balanceOf(user);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);

    const contract = ast_file.item(ast_file.root_items[2]).Contract;
    const function = ast_file.item(contract.members[1]).Function;
    const decl = ast_file.statement(ast_file.body(function.body).statements[0]).VariableDecl;
    const result_pattern = findVariablePatternByName(ast_file, ast_file.body(function.body).statements, "call_result").?;
    const result_type = typecheck.pattern_types[result_pattern.index()].type;
    _ = decl;
    try testing.expectEqual(compiler.sema.TypeKind.error_union, result_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, result_type.payloadType().?.kind());
    try testing.expectEqualStrings("ExternalCallFailed", result_type.errorTypes()[0].named.name);
}

test "compiler includes declared extern trait errors in call result types" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(InsufficientBalance, InvalidRecipient);
        \\}
        \\
        \\error ExternalCallFailed;
        \\error InsufficientBalance;
        \\error InvalidRecipient;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn send(to: address, amount: u256) {
        \\        let call_result = external<ERC20>(token, gas: 50000).transfer(to, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);

    const contract = ast_file.item(ast_file.root_items[4]).Contract;
    const function = ast_file.item(contract.members[1]).Function;
    const result_pattern = findVariablePatternByName(ast_file, ast_file.body(function.body).statements, "call_result").?;
    const result_type = typecheck.pattern_types[result_pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.error_union, result_type.kind());
    try testing.expectEqual(@as(usize, 3), result_type.errorTypes().len);
    try testing.expectEqualStrings("ExternalCallFailed", result_type.errorTypes()[0].named.name);
    try testing.expectEqualStrings("InsufficientBalance", result_type.errorTypes()[1].named.name);
    try testing.expectEqualStrings("InvalidRecipient", result_type.errorTypes()[2].named.name);
}

test "compiler rejects unknown extern trait errors clauses" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(UnknownError);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "extern trait method 'transfer' declares unknown error 'UnknownError'"));
}

test "compiler accepts payload-bearing extern trait errors clauses" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(InsufficientBalance);
        \\}
        \\
        \\error InsufficientBalance(required: u256, available: u256);
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler reports external proxy misuse" {
    const source_text =
        \\trait Plain {
        \\    fn ping(self) -> bool;
        \\}
        \\
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn badMissingGas(user: address) -> !u256 | ExternalCallFailed {
        \\        return external<ERC20>(token).balanceOf(user);
        \\    }
        \\
        \\    pub fn badTrait() -> !bool | ExternalCallFailed {
        \\        return external<Plain>(token, gas: 50000).ping();
        \\    }
        \\
        \\    pub fn badMethod(user: address) -> !u256 | ExternalCallFailed {
        \\        return external<ERC20>(token, gas: 50000).missing(user);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const syntax_diags = try compilation.db.syntaxDiagnostics(compilation.db.sources.module(compilation.root_module_id).file_id);
    try testing.expect(diagnosticMessagesContain(syntax_diags, "expected ', gas: ...' in external proxy"));

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "trait 'Plain' is not extern"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "type 'external proxy' has no field 'missing'"));
}

test "compiler rejects storage write after extern call on same slot" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\    storage var other: u256;
        \\
        \\    pub fn bad(to: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).transfer(to, balance);
        \\        _ = call_result;
        \\        balance = 2;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write storage slot 'balance' after external call because it was written before the call"));
}

test "compiler allows writes to different storage slots around extern call" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\    storage var other: u256;
        \\
        \\    pub fn ok(to: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).transfer(to, balance);
        \\        _ = call_result;
        \\        other = 2;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler allows writes around extern staticcall" {
    const source_text =
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\
        \\    pub fn ok(user: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).balanceOf(user);
        \\        _ = call_result;
        \\        balance = 2;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler allows post-call writes when pre-call storage write is branch-local" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Test {
        \\    storage var balance: u256 = 0;
        \\    storage var token: address;
        \\
        \\    pub fn example(flag: bool, addr: address) {
        \\        if (flag) {
        \\            balance = 100;
        \\        }
        \\        let ok = try external<ERC20>(token, gas: 50000).transfer(addr, 1);
        \\        _ = ok;
        \\        balance += 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler rejects post-call writes when all branches wrote same storage slot before call" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Test {
        \\    storage var balance: u256 = 0;
        \\    storage var token: address;
        \\
        \\    pub fn example(flag: bool, addr: address) {
        \\        if (flag) {
        \\            balance = 100;
        \\        } else {
        \\            balance = 200;
        \\        }
        \\        let ok = try external<ERC20>(token, gas: 50000).transfer(addr, 1);
        \\        _ = ok;
        \\        balance += 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write storage slot 'balance' after external call because it was written before the call"));
}

test "compiler still rejects same-slot write before and after extern call without branches" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Test {
        \\    storage var balance: u256 = 0;
        \\    storage var token: address;
        \\
        \\    pub fn example(addr: address) {
        \\        balance = 100;
        \\        let ok = try external<ERC20>(token, gas: 50000).transfer(addr, 1);
        \\        _ = ok;
        \\        balance += 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write storage slot 'balance' after external call because it was written before the call"));
}

test "compiler rejects unknown error returns" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Nonexistent(7);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const resolution_diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(resolution_diags, "undefined name 'Nonexistent'"));
}

test "compiler rejects error returns with wrong payload arity" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Failure(1, 2, 3);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "expected 1 arguments, found 3"));
}

test "compiler rejects error returns outside function return error set" {
    const source_text =
        \\error ErrorA;
        \\error ErrorB;
        \\
        \\pub fn run() -> !u256 | ErrorA {
        \\    return ErrorB();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "return expects type 'error union', found 'ErrorB'"));
}

test "compiler rejects error returns with wrong payload types" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Failure(true, 7);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.items.items.len != 0);
}

test "compiler widens narrower error unions into wider return sets" {
    const source_text =
        \\error ErrorA;
        \\error ErrorB;
        \\
        \\fn narrow(maybe: !u256 | ErrorA) -> !u256 | ErrorA {
        \\    return maybe;
        \\}
        \\
        \\pub fn wide(maybe: !u256 | ErrorA) -> !u256 | ErrorA | ErrorB {
        \\    return narrow(maybe);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler widens narrower error unions through try expressions" {
    const source_text =
        \\error ErrorA;
        \\error ErrorB;
        \\
        \\fn narrow(maybe: !u256 | ErrorA) -> !u256 | ErrorA {
        \\    return maybe;
        \\}
        \\
        \\pub fn wide(maybe: !u256 | ErrorA) -> !u256 | ErrorA | ErrorB {
        \\    return try narrow(maybe);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler types single-error catch bindings as concrete error payloads" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn handle(maybe: !u256 | Failure) -> u256 {
        \\    try {
        \\        maybe;
        \\    } catch (e) {
        \\        return e.code;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const try_stmt = ast_file.statement(body.statements[0]).Try;
    const catch_clause = try_stmt.catch_clause.?;
    const catch_pattern = catch_clause.error_pattern.?;
    const catch_body = ast_file.body(catch_clause.body);
    const ret = ast_file.statement(catch_body.statements[0]).Return;
    const field_expr = ret.value.?;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
    try testing.expectEqual(compiler.sema.TypeKind.named, typecheck.pattern_types[catch_pattern.index()].kind());
    try testing.expectEqualStrings("Failure", typecheck.pattern_types[catch_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(field_expr).kind());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
}

test "compiler rejects field access on multi-error catch bindings" {
    const source_text =
        \\error ErrorA(code: u256);
        \\error ErrorB(required: u256);
        \\
        \\pub fn handle(maybe: !u256 | ErrorA | ErrorB) -> u256 {
        \\    try {
        \\        maybe;
        \\    } catch (e) {
        \\        return e.code;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "catch binding represents multiple possible error types; field access is not supported"));
}

test "compiler types multi-field single-error catch bindings" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\
        \\pub fn handle(maybe: !u256 | Failure) -> address {
        \\    try {
        \\        maybe;
        \\    } catch (e) {
        \\        return e.owner;
        \\    }
        \\    return 0x0000000000000000000000000000000000000000;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const try_stmt = ast_file.statement(body.statements[0]).Try;
    const catch_clause = try_stmt.catch_clause.?;
    const catch_pattern = catch_clause.error_pattern.?;
    const catch_body = ast_file.body(catch_clause.body);
    const ret = ast_file.statement(catch_body.statements[0]).Return;
    const field_expr = ret.value.?;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
    try testing.expectEqual(compiler.sema.TypeKind.named, typecheck.pattern_types[catch_pattern.index()].kind());
    try testing.expectEqualStrings("Failure", typecheck.pattern_types[catch_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.address, typecheck.exprType(field_expr).kind());
}

test "compiler lowers extern trait calls to abi and external call ops" {
    const source_text =
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn probe(user: address) -> !u256 | ExternalCallFailed {
        \\        return external<ERC20>(token, gas: 50000).balanceOf(user);
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.abi_encode"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.external_call"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.abi_decode"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"staticcall\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"ERC20\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"balanceOf\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.ok"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.err"));
}

test "compiler lowers zero-payload extern trait errors clauses into selector matching" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(InsufficientBalance, InvalidRecipient);
        \\}
        \\
        \\error ExternalCallFailed;
        \\error InsufficientBalance;
        \\error InvalidRecipient;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn send(to: address, amount: u256) -> !bool | ExternalCallFailed | InsufficientBalance | InvalidRecipient {
        \\        return external<ERC20>(token, gas: 50000).transfer(to, amount);
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.abi_decode"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @InsufficientBalance"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @InvalidRecipient"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @ExternalCallFailed"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "scf.if"));
}

test "compiler lowers payload-bearing extern trait errors into selector matching and decode" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(InsufficientBalance);
        \\}
        \\
        \\error ExternalCallFailed;
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn send(to: address, amount: u256) -> !bool | ExternalCallFailed | InsufficientBalance {
        \\        return external<ERC20>(token, gas: 50000).transfer(to, amount);
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 3, "ora.abi_decode"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @ExternalCallFailed"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "call @InsufficientBalance"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.addi"));
}

test "compiler converts extern trait calls through SIR" {
    const source_text =
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn probe(user: address) -> !u256 | ExternalCallFailed {
        \\        return external<ERC20>(token, gas: 50000).balanceOf(user);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.staticcall"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.trait_name"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"ERC20\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.method_name"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"balanceOf\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.selector"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
}

test "compiler converts payload-bearing extern trait errors through SIR" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(InsufficientBalance);
        \\}
        \\
        \\error ExternalCallFailed;
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn send(to: address, amount: u256) -> !bool | ExternalCallFailed | InsufficientBalance {
        \\        return external<ERC20>(token, gas: 50000).transfer(to, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.returndatacopy"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.malloc"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts call-kind extern traits with bool and address returns through SIR" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\    staticcall fn owner(self) -> address;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn send(to: address, amount: u256) -> !bool | ExternalCallFailed {
        \\        return external<ERC20>(token, gas: 50000).transfer(to, amount);
        \\    }
        \\
        \\    pub fn currentOwner() -> !address | ExternalCallFailed {
        \\        return external<ERC20>(token, gas: 50000).owner();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.call"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.staticcall"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.call_kind"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"call\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"staticcall\""));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
}

test "compiler converts extern trait calls with narrow integer returns through SIR" {
    const source_text =
        \\extern trait ERC20 {
        \\    staticcall fn decimals(self) -> u8;
        \\    staticcall fn basisPoints(self) -> u16;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn tokenDecimals() -> !u8 | ExternalCallFailed {
        \\        return external<ERC20>(token, gas: 50000).decimals();
        \\    }
        \\
        \\    pub fn feeBps() -> !u16 | ExternalCallFailed {
        \\        return external<ERC20>(token, gas: 50000).basisPoints();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.staticcall"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
}

test "compiler converts extern trait calls with dynamic bytes and string returns through SIR" {
    const source_text =
        \\extern trait ERC20Meta {
        \\    staticcall fn name(self) -> string;
        \\    staticcall fn symbolBytes(self) -> bytes;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn tokenName() -> !string | ExternalCallFailed {
        \\        return external<ERC20Meta>(token, gas: 50000).name();
        \\    }
        \\
        \\    pub fn tokenSymbolBytes() -> !bytes | ExternalCallFailed {
        \\        return external<ERC20Meta>(token, gas: 50000).symbolBytes();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.staticcall"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.returndatasize"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.returndatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
}

test "compiler converts extern trait calls with static struct returns through SIR" {
    const source_text =
        \\struct Snapshot {
        \\    owner: address;
        \\    amount: u256;
        \\}
        \\
        \\extern trait VaultView {
        \\    staticcall fn snapshot(self) -> Snapshot;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var target: address;
        \\
        \\    pub fn snapshotView() -> !Snapshot | ExternalCallFailed {
        \\        return external<VaultView>(target, gas: 50000).snapshot();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.staticcall"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.returndatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
}

test "compiler converts extern trait calls with tuple returns through SIR" {
    const source_text =
        \\extern trait VaultView {
        \\    staticcall fn quote(self) -> (u256, bool);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var target: address;
        \\
        \\    pub fn quoteView() -> !(u256, bool) | ExternalCallFailed {
        \\        return external<VaultView>(target, gas: 50000).quote();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.staticcall"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.returndatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
}

test "compiler computes extern trait ABI signatures" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const trait_interface = typecheck.traitInterfaceByName("ERC20").?;

    const transfer_signature = try compiler.hir.abi.signatureForMethod(
        testing.allocator,
        trait_interface.methods[0].name,
        trait_interface.methods[0].has_self,
        trait_interface.methods[0].param_types,
    );
    defer testing.allocator.free(transfer_signature);
    try testing.expectEqualStrings("transfer(address,uint256)", transfer_signature);

    const balance_signature = try compiler.hir.abi.signatureForMethod(
        testing.allocator,
        trait_interface.methods[1].name,
        trait_interface.methods[1].has_self,
        trait_interface.methods[1].param_types,
    );
    defer testing.allocator.free(balance_signature);
    try testing.expectEqualStrings("balanceOf(address)", balance_signature);
}

test "compiler computes extern trait selectors" {
    const selector = try compiler.hir.abi.keccakSelectorHex(testing.allocator, "transfer(address,uint256)");
    defer testing.allocator.free(selector);
    try testing.expectEqualStrings("0xa9059cbb", selector);
}

test "compiler abi generation uses compiler pipeline for public abi" {
    const source_text =
        \\contract Test {
        \\    storage var counter: u256;
        \\    error InvalidAmount(amount: u256);
        \\    log Transfer(indexed from: address, amount: u256);
        \\    pub fn viewFn() -> u256 { return counter; }
        \\    pub fn writeFn(amount: u256) { counter = amount; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    try testing.expectEqual(@as(usize, 1), contract_abi.contract_count);
    try testing.expectEqualStrings("Test", contract_abi.contract_name);

    var saw_view = false;
    var saw_write = false;
    var saw_error = false;
    var saw_event = false;

    for (contract_abi.callables) |callable| {
        if (callable.kind == .function and std.mem.eql(u8, callable.name, "viewFn")) {
            saw_view = true;
            try testing.expect(callable.selector != null);
        }
        if (callable.kind == .function and std.mem.eql(u8, callable.name, "writeFn")) {
            saw_write = true;
            try testing.expect(callable.selector != null);
        }
        if (callable.kind == .@"error" and std.mem.eql(u8, callable.name, "InvalidAmount")) {
            saw_error = true;
            try testing.expectEqual(@as(usize, 1), callable.inputs.len);
        }
        if (callable.kind == .event and std.mem.eql(u8, callable.name, "Transfer")) {
            saw_event = true;
            try testing.expect(callable.selector == null);
        }
    }

    try testing.expect(saw_view);
    try testing.expect(saw_write);
    try testing.expect(saw_error);
    try testing.expect(saw_event);
}

test "compiler preserves trait ghost blocks in AST" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\
        \\    ghost {
        \\        assert(true, "ok");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const trait_item = ast_file.item(ast_file.root_items[0]).Trait;

    try testing.expect(trait_item.ghost_block != null);
    const ghost_item = ast_file.item(trait_item.ghost_block.?).GhostBlock;
    const body = ast_file.body(ghost_item.body).*;
    try testing.expectEqual(@as(usize, 1), body.statements.len);
}

test "compiler collects verification facts from trait ghost blocks" {
    const source_text =
        \\trait SafeCounter {
        \\    fn get(self) -> u256;
        \\
        \\    ghost {
        \\        assume(true);
        \\        assert(get(self) >= 0, "non-negative");
        \\        get(self) >= 0;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const trait_id = ast_file.root_items[0];

    const facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = trait_id });
    try testing.expectEqual(@as(usize, 3), facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, facts.facts[1].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.invariant, facts.facts[2].kind);
}

test "compiler type-checks trait ghost blocks during impl checking" {
    const source_text =
        \\trait SafeCounter {
        \\    fn get(self) -> u256;
        \\
        \\    ghost {
        \\        assert(1, "bad");
        \\    }
        \\}
        \\
        \\contract Counter {}
        \\
        \\impl SafeCounter for Counter {
        \\    fn get(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "assert condition must be 'bool'"));
}

test "compiler lowers trait ghost blocks into verification HIR" {
    const source_text =
        \\trait SafeCounter {
        \\    fn get(self) -> u256;
        \\
        \\    ghost {
        \\        assert(true, "safe");
        \\    }
        \\}
        \\
        \\contract Counter {}
        \\
        \\impl SafeCounter for Counter {
        \\    fn get(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Counter.get"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));
}

test "compiler verifies impls with trait ghost blocks end to end" {
    const source_text =
        \\trait SafeCounter {
        \\    fn get(self) -> u256;
        \\
        \\    ghost {
        \\        assert(true, "safe");
        \\    }
        \\}
        \\
        \\contract Counter {}
        \\
        \\impl SafeCounter for Counter {
        \\    fn get(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Counter.get"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));

    const z3_verification = @import("ora_z3_verification");
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.parallel = false;

    var result = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "compiler verifies trait ghost method calls with self end to end" {
    const source_text =
        \\trait SafeCounter {
        \\    fn get(self) -> u256;
        \\
        \\    ghost {
        \\        assert(get(self) >= 0, "safe");
        \\    }
        \\}
        \\
        \\contract Counter {}
        \\
        \\impl SafeCounter for Counter {
        \\    fn get(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const resolution = try compilation.db.resolveNames(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), resolution.diagnostics.items.items.len);

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Counter.get"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));

    const z3_verification = @import("ora_z3_verification");
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.parallel = false;

    var result = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "compiler lowers ensures on implicit void returns" {
    const source_text =
        \\contract Counter {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn bump() ensures total >= 1 {
        \\        total = 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    const ensures_index = std.mem.indexOf(u8, hir_text, "ora.ensures") orelse return error.TestUnexpectedResult;
    const return_index = std.mem.indexOf(u8, hir_text, "ora.return") orelse return error.TestUnexpectedResult;
    try testing.expect(ensures_index < return_index);
}

test "compiler reports trait method body parse error" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(diags, "trait methods cannot have a body"));
}

test "compiler reports non-method elements in trait bodies clearly" {
    const source_text =
        \\trait ERC20 {
        \\    let value = 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(diags, "expected method signature in trait body"));
}

test "compiler reports impl syntax errors for missing body and missing for" {
    const source_text =
        \\impl ERC20 Token {
        \\    fn totalSupply(self) -> u256;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(diags, "expected 'for' in impl declaration"));
    try testing.expect(diagnosticMessagesContain(diags, "impl methods must have a body"));
}

test "compiler indexes traits and impls by trait target pair" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);

    const trait_item_id = item_index.lookup("ERC20");
    try testing.expect(trait_item_id != null);
    try testing.expect(ast_file.item(trait_item_id.?).* == .Trait);

    const impl_item_id = item_index.lookupImpl("ERC20", "Token");
    try testing.expect(impl_item_id != null);
    try testing.expect(ast_file.item(impl_item_id.?).* == .Impl);
}

test "compiler allows bare self in trait and impl methods" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const syntax_diags = try compilation.db.syntaxDiagnostics(module.file_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(!diagnosticMessagesContain(syntax_diags, "bare 'self'"));
    try testing.expect(!diagnosticMessagesContain(ast_diags, "bare 'self'"));
}

test "compiler rejects bare self in ordinary functions" {
    const source_text =
        \\pub fn bad(self) -> u256 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(diagnosticMessagesContain(ast_diags, "bare 'self' parameter is only allowed in trait and impl methods"));
}

test "compiler type checks valid trait impl conformance" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\    fn transfer(self, to: address, amount: u256) -> bool;
        \\    fn decimals() -> u8;
        \\}
        \\
        \\contract Token {}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\    fn transfer(self, to: address, amount: u256) -> bool {
        \\        _ = to;
        \\        _ = amount;
        \\        return true;
        \\    }
        \\    fn decimals() -> u8 { return 18; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);
}

test "compiler reports missing and extra trait impl methods" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\    fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\contract Token {}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\    fn extra(self) -> u256 { return 1; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "impl missing method 'transfer' required by trait 'ERC20'"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "impl contains method 'extra' which is not part of trait 'ERC20'"));
}

test "compiler reports wrong trait impl parameter and return signatures" {
    const source_text =
        \\trait ERC20 {
        \\    fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\contract Token {}
        \\
        \\impl ERC20 for Token {
        \\    fn transfer(self, to: bool, amount: u256) -> u256 {
        \\        _ = to;
        \\        _ = amount;
        \\        return 0;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "method 'transfer' has wrong signature for trait 'ERC20': parameter 0 expects 'address', found 'bool'"));
}

test "compiler reports trait impl return signature mismatch" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\}
        \\
        \\contract Token {}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> bool { return true; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "method 'totalSupply' has wrong signature for trait 'ERC20': expected return 'u256', found 'bool'"));
}

test "compiler reports duplicate impl for same trait and target" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\}
        \\
        \\contract Token {}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 1; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "duplicate impl for trait 'ERC20' and type 'Token'"));
}

test "compiler exposes trait and impl interfaces in sema" {
    const source_text =
        \\trait ERC20 {
        \\    fn totalSupply(self) -> u256;
        \\    fn decimals() -> u8;
        \\}
        \\
        \\contract Token {}
        \\
        \\impl ERC20 for Token {
        \\    fn totalSupply(self) -> u256 { return 0; }
        \\    fn decimals() -> u8 { return 18; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const trait_interface = typecheck.traitInterfaceByName("ERC20");
    try testing.expect(trait_interface != null);
    try testing.expectEqual(@as(usize, 2), trait_interface.?.methods.len);
    try testing.expectEqualStrings("totalSupply", trait_interface.?.methods[0].name);
    try testing.expect(trait_interface.?.methods[0].has_self);
    try testing.expectEqual(compiler.sema.TypeKind.integer, trait_interface.?.methods[0].return_type.kind());
    try testing.expectEqualStrings("decimals", trait_interface.?.methods[1].name);
    try testing.expect(!trait_interface.?.methods[1].has_self);

    const impl_interface = typecheck.implInterfaceByNames("ERC20", "Token");
    try testing.expect(impl_interface != null);
    try testing.expectEqual(@as(usize, 2), impl_interface.?.methods.len);
    try testing.expectEqualStrings("totalSupply", impl_interface.?.methods[0].name);
    try testing.expect(impl_interface.?.methods[0].has_self);
    try testing.expectEqualStrings("decimals", impl_interface.?.methods[1].name);
}

test "compiler parses and lowers trait bounds on generic functions" {
    const source_text =
        \\trait Comparable {
        \\    fn compare(self, other: u256) -> bool;
        \\}
        \\
        \\fn keep(comptime T: type, value: T) -> T where T: Comparable, T: Comparable {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const syntax_tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(syntax_tree);
    try testing.expect(containsNodeOfKind(root, .TraitBoundClause));

    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    try testing.expectEqual(@as(usize, 2), function.trait_bounds.len);
    try testing.expectEqualStrings("T", function.trait_bounds[0].parameter_name);
    try testing.expectEqualStrings("Comparable", function.trait_bounds[0].trait_name);
}

test "compiler accepts bounded generic calls for implemented trait types" {
    const source_text =
        \\trait Marker {
        \\    fn marked(self) -> bool;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    fn marked(self) -> bool {
        \\        return true;
        \\    }
        \\}
        \\
        \\contract Test {
        \\    fn keep(comptime T: type, value: T) -> T where T: Marker {
        \\        return value;
        \\    }
        \\
        \\    pub fn run(value: Box) -> Box {
        \\        return keep(Box, value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler infers generic type arguments from runtime call arguments" {
    const source_text =
        \\contract Test {
        \\    fn add(comptime T: type, a: T, b: T) -> T {
        \\        return a + b;
        \\    }
        \\
        \\    pub fn run(a: u256, b: u256) -> u256 {
        \\        return add(a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "@add__"));
}

test "compiler still accepts explicit generic type arguments" {
    const source_text =
        \\contract Test {
        \\    fn add(comptime T: type, a: T, b: T) -> T {
        \\        return a + b;
        \\    }
        \\
        \\    pub fn run(a: u256, b: u256) -> u256 {
        \\        return add(u256, a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "@add__"));
}

test "compiler accepts explicit comptime value bindings on generic calls" {
    const source_text =
        \\contract Test {
        \\    fn ct(comptime T: type, comptime value: T) -> T {
        \\        return value;
        \\    }
        \\
        \\    pub fn run() -> u8 {
        \\        return ct(u8, 18);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.constant 18"));
}

test "compiler rejects conflicting inferred generic type arguments" {
    const source_text =
        \\contract Test {
        \\    fn pick(comptime T: type, a: T, b: T) -> T {
        \\        return a;
        \\    }
        \\
        \\    pub fn run(a: u256, b: bool) -> u256 {
        \\        return pick(a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "could not infer generic type arguments"));
}

test "compiler rejects bounded generic calls for types without impl" {
    const source_text =
        \\trait Marker {
        \\    fn marked(self) -> bool;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\struct Plain {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    fn marked(self) -> bool {
        \\        return true;
        \\    }
        \\}
        \\
        \\contract Test {
        \\    fn keep(comptime T: type, value: T) -> T where T: Marker {
        \\        return value;
        \\    }
        \\
        \\    pub fn run(value: Plain) -> Plain {
        \\        return keep(Plain, value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "type 'Plain' does not implement trait 'Marker'"));
}

test "compiler resolves trait-bound methods in generic bodies" {
    const source_text =
        \\trait Marker {
        \\    fn marked(self) -> bool;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    fn marked(self) -> bool {
        \\        return self.value > 0;
        \\    }
        \\}
        \\
        \\contract Test {
        \\    fn choose(comptime T: type, a: T) -> bool where T: Marker {
        \\        return a.marked();
        \\    }
        \\
        \\    pub fn run(a: Box) -> bool {
        \\        return choose(Box, a);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("Test").?;
    const contract = ast_file.item(contract_id).Contract;
    var choose_id: ?compiler.ast.ItemId = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;
        if (std.mem.eql(u8, item.Function.name, "choose")) {
            choose_id = member_id;
            break;
        }
    }
    try testing.expect(choose_id != null);
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.itemLocatedType(choose_id.?).type.function.return_types[0].kind());

    const choose_fn = ast_file.item(choose_id.?).Function;
    const body = ast_file.body(choose_fn.body).*;
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const call_expr = ret_stmt.value.?;
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(call_expr).kind());
}

test "compiler lowers trait-bound generic method calls to concrete impl symbols" {
    const source_text =
        \\trait Marker {
        \\    fn marked(self) -> bool;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    fn marked(self) -> bool {
        \\        return self.value > 0;
        \\    }
        \\}
        \\
        \\contract Test {
        \\    fn choose(comptime T: type, a: T) -> bool where T: Marker {
        \\        return a.marked();
        \\    }
        \\
        \\    pub fn run(a: Box) -> bool {
        \\        return choose(Box, a);
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Box.marked"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @choose__Box"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @run"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @Box.marked"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @choose__Box"));
}

test "compiler types bare self in impl bodies as the impl target" {
    const source_text =
        \\trait Marker {
        \\    fn marked(self) -> bool;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    fn marked(self) -> bool {
        \\        return self.value > 0;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const impl_id = item_index.lookupImpl("Marker", "Box").?;
    const impl_item = ast_file.item(impl_id).Impl;
    const method_id = impl_item.methods[0];
    const method = ast_file.item(method_id).Function;
    try testing.expectEqualStrings("Box", typecheck.itemLocatedType(method_id).type.function.param_types[0].name().?);
    try testing.expectEqualStrings("Box", typecheck.pattern_types[method.parameters[0].pattern.index()].type.name().?);
}

test "compiler lowers generic impl methods for trait-bound calls" {
    const source_text =
        \\trait Marker {
        \\    fn marked(comptime N: u256, self) -> u256;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    fn marked(comptime N: u256, self) -> u256 {
        \\        return N;
        \\    }
        \\}
        \\
        \\contract Test {
        \\    fn choose(comptime T: type, a: T) -> u256 where T: Marker {
        \\        return a.marked(7);
        \\    }
        \\
        \\    pub fn run(a: Box) -> u256 {
        \\        return choose(Box, a);
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Box.marked__"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @Box.marked__"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @choose__Box"));
}

test "compiler type checks associated trait calls in generic bodies" {
    const source_text =
        \\trait Factory {
        \\    fn make() -> bool;
        \\}
        \\
        \\struct Box {}
        \\
        \\impl Factory for Box {
        \\    fn make() -> bool {
        \\        return true;
        \\    }
        \\}
        \\
        \\contract Test {
        \\    fn choose(comptime T: type) -> bool where T: Factory {
        \\        return T.make();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("Test").?;
    const contract = ast_file.item(contract_id).Contract;
    var choose_id: ?compiler.ast.ItemId = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;
        if (std.mem.eql(u8, item.Function.name, "choose")) {
            choose_id = member_id;
            break;
        }
    }
    try testing.expect(choose_id != null);
    const choose_fn = ast_file.item(choose_id.?).Function;
    const body = ast_file.body(choose_fn.body).*;
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const call_expr = ret_stmt.value.?;
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(call_expr).kind());
}

test "compiler reports missing trait bounds for trait method calls" {
    const source_text =
        \\trait Marker {
        \\    fn marked(self) -> bool;
        \\}
        \\
        \\fn choose(comptime T: type, a: T) -> bool {
        \\    return a.marked();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "type parameter 'T' has no trait bound providing method 'marked'"));
}

test "compiler reports missing impls for concrete associated trait calls" {
    const source_text =
        \\trait Factory {
        \\    fn make() -> bool;
        \\}
        \\
        \\struct Box {}
        \\
        \\pub fn run() -> bool {
        \\    return Box.make();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "type 'Box' has no impl providing method 'make'"));
}

test "compiler reports ambiguous trait method names across impls" {
    const source_text =
        \\trait Left {
        \\    fn mark() -> bool;
        \\}
        \\
        \\trait Right {
        \\    fn mark() -> bool;
        \\}
        \\
        \\struct Box {}
        \\
        \\impl Left for Box {
        \\    fn mark() -> bool { return true; }
        \\}
        \\
        \\impl Right for Box {
        \\    fn mark() -> bool { return false; }
        \\}
        \\
        \\fn choose() -> bool {
        \\    return Box.mark();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "method 'mark' is ambiguous for type 'Box' across multiple impls"));
}

test "compiler lowers associated trait impl calls to concrete symbols" {
    const source_text =
        \\trait Factory {
        \\    fn make() -> bool;
        \\}
        \\
        \\struct Box {}
        \\
        \\impl Factory for Box {
        \\    fn make() -> bool {
        \\        return true;
        \\    }
        \\}
        \\
        \\contract Test {
        \\    fn choose(comptime T: type) -> bool where T: Factory {
        \\        return T.make();
        \\    }
        \\
        \\    pub fn run() -> bool {
        \\        return choose(Box);
        \\    }
        \\
        \\    pub fn direct() -> bool {
        \\        return Box.make();
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Box.make"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @choose__Box"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "call @Box.make"));
}

test "compiler syntax parses expression precedence and postfix chains" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    foo.bar(1 + 2 * 3, xs[4]).baz;
        \\    total = a + b * c;
        \\    let value = left + right * other;
        \\    assert(value >= 0, "ok");
        \\    return (left + right) * value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = nthChildNodeOfKind(root, .FunctionItem, 0).?;
    const body = nthChildNodeOfKind(function, .Body, 0).?;

    const expr_stmt = nthChildNodeOfKind(body, .ExprStmt, 0).?;
    const call_chain = nthChildNodeOfKind(expr_stmt, .FieldExpr, 0);
    try testing.expect(call_chain != null);
    try testing.expect(containsNodeOfKind(expr_stmt, .CallExpr));
    try testing.expect(containsNodeOfKind(expr_stmt, .IndexExpr));

    const assign_stmt = nthChildNodeOfKind(body, .AssignStmt, 0).?;
    const assign_rhs = nthChildNodeOfKind(assign_stmt, .BinaryExpr, 0);
    try testing.expect(assign_rhs != null);
    try testing.expect(containsNodeOfKind(assign_rhs.?, .BinaryExpr));

    const var_stmt = nthChildNodeOfKind(body, .VariableDeclStmt, 0).?;
    try testing.expect(containsNodeOfKind(var_stmt, .BinaryExpr));

    const assert_stmt = nthChildNodeOfKind(body, .AssertStmt, 0).?;
    try testing.expect(containsNodeOfKind(assert_stmt, .BinaryExpr));

    const return_stmt = nthChildNodeOfKind(body, .ReturnStmt, 0).?;
    try testing.expect(containsNodeOfKind(return_stmt, .BinaryExpr));
    try testing.expect(containsNodeOfKind(return_stmt, .GroupExpr));
}

test "compiler syntax parses control-flow conditions and structured expression contexts" {
    const source_text =
        \\pub fn run(items: u256, value: u256) -> u256 {
        \\    if (a + b > c) {
        \\        while (value > 0)
        \\            invariant value + 1 > 0;
        \\        {
        \\            break;
        \\        }
        \\    }
        \\
        \\    switch (value + 1) {
        \\        0 => 1;
        \\        1...2 => Point{ x: left + 1, y: right };
        \\        else => 3;
        \\    }
        \\
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = nthChildNodeOfKind(root, .FunctionItem, 0).?;
    const body = nthChildNodeOfKind(function, .Body, 0).?;

    const if_stmt = nthChildNodeOfKind(body, .IfStmt, 0).?;
    try testing.expect(containsNodeOfKind(if_stmt, .BinaryExpr));
    try testing.expect(firstChildNodeOfKind(if_stmt, .GroupParen) == null);

    const if_body = nthChildNodeOfKind(if_stmt, .Body, 0).?;
    const while_stmt = nthChildNodeOfKind(if_body, .WhileStmt, 0).?;
    const invariant = nthChildNodeOfKind(while_stmt, .InvariantClause, 0).?;
    try testing.expect(containsNodeOfKind(while_stmt, .BinaryExpr));
    try testing.expect(containsNodeOfKind(invariant, .BinaryExpr));

    const switch_stmt = nthChildNodeOfKind(body, .SwitchStmt, 0).?;
    try testing.expect(containsNodeOfKind(switch_stmt, .BinaryExpr));
    try testing.expect(firstChildNodeOfKind(switch_stmt, .GroupParen) == null);

    const range_arm = nthChildNodeOfKind(switch_stmt, .SwitchArm, 1).?;
    try testing.expect(containsNodeOfKind(range_arm, .RangeExpr));
    try testing.expect(containsNodeOfKind(range_arm, .StructLiteral));
    try testing.expect(containsNodeOfKind(range_arm, .AnonymousStructLiteralField));
    try testing.expect(containsNodeOfKind(range_arm, .BinaryExpr));
}

test "compiler syntax parses type expressions in signatures and declarations" {
    const source_text =
        \\struct Pair {
        \\    left: map<u256, address>,
        \\    right: (u256, bool),
        \\}
        \\
        \\bitfield Flags: !Result | Error {
        \\    enabled: slice[address] @bits(0..1);
        \\}
        \\
        \\pub fn run(
        \\    values: map<u256, address>,
        \\    pair: (u256, bool),
        \\    items: [u256; 4],
        \\    addrs: slice[address],
        \\) -> !map<u256, address> | Error {
        \\    let local: struct { x: u256, y: bool } = Point{ x: 1, y: true };
        \\    return values;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    const struct_item = nthChildNodeOfKind(root, .StructItem, 0).?;
    try testing.expect(containsNodeOfKind(struct_item, .StructField));
    try testing.expect(containsNodeOfKind(struct_item, .GenericType));
    try testing.expect(containsNodeOfKind(struct_item, .TupleType));

    const bitfield_item = nthChildNodeOfKind(root, .BitfieldItem, 0).?;
    try testing.expect(containsNodeOfKind(bitfield_item, .ErrorUnionType));
    try testing.expect(containsNodeOfKind(bitfield_item, .SliceType));

    const function = nthChildNodeOfKind(root, .FunctionItem, 0).?;
    const params = nthChildNodeOfKind(function, .ParameterList, 0).?;
    try testing.expect(nthChildNodeOfKind(params, .Parameter, 0) != null);
    try testing.expect(nthChildNodeOfKind(params, .Parameter, 3) != null);
    try testing.expect(containsNodeOfKind(params, .GenericType));
    try testing.expect(containsNodeOfKind(params, .TupleType));
    try testing.expect(containsNodeOfKind(params, .ArrayType));
    try testing.expect(containsNodeOfKind(params, .SliceType));
    try testing.expect(containsNodeOfKind(function, .ErrorUnionType));

    const body = nthChildNodeOfKind(function, .Body, 0).?;
    const local_decl = nthChildNodeOfKind(body, .VariableDeclStmt, 0).?;
    try testing.expect(containsNodeOfKind(local_decl, .AnonymousStructType));
    try testing.expect(containsNodeOfKind(local_decl, .AnonymousStructField));
}

test "compiler lowers syntax into immutable AST items" {
    const source_text =
        \\contract Counter {
        \\    storage var value: u256;
        \\    invariant value >= 0;
        \\
        \\    pub fn get() -> u256 {
        \\        return value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const file = try compilation.db.astFile(module.file_id);

    try testing.expectEqual(@as(usize, 1), file.root_items.len);
    try testing.expect(file.item(file.root_items[0]).* == .Contract);
    try testing.expect(file.item(file.root_items[0]).Contract.members.len >= 2);
}

test "compiler parses and lowers structured top-level item forms" {
    const source_text =
        \\comptime const math = @import("./math.ora");
        \\storage var total: u256 = 1;
        \\const LIMIT: u256 = 2;
        \\log Transfer(indexed from: address, to: address, amount: u256);
        \\error Failed(code: u256);
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    try testing.expect(nthChildNodeOfKind(root, .ImportItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .FieldItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .ConstantItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .LogDeclItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .ErrorDeclItem, 0) != null);

    const log_decl = nthChildNodeOfKind(root, .LogDeclItem, 0).?;
    try testing.expect(nthChildNodeOfKind(log_decl, .LogField, 0) != null);

    const error_decl = nthChildNodeOfKind(root, .ErrorDeclItem, 0).?;
    try testing.expect(nthChildNodeOfKind(error_decl, .ParameterList, 0) != null);

    const file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 5), file.root_items.len);

    var found_import = false;
    var found_field = false;
    var found_constant = false;
    var found_log = false;
    var found_error = false;

    for (file.root_items) |item_id| {
        switch (file.item(item_id).*) {
            .Import => |import_item| {
                found_import = true;
                try testing.expect(import_item.is_comptime);
                try testing.expectEqualStrings("math", import_item.alias.?);
                try testing.expectEqualStrings("./math.ora", import_item.path);
            },
            .Field => |field_item| {
                found_field = true;
                try testing.expectEqual(.storage, field_item.storage_class);
                try testing.expectEqual(.var_, field_item.binding_kind);
                try testing.expect(field_item.type_expr != null);
                try testing.expect(field_item.value != null);
            },
            .Constant => |constant_item| {
                if (std.mem.eql(u8, constant_item.name, "LIMIT")) {
                    found_constant = true;
                    try testing.expect(!constant_item.is_comptime);
                    try testing.expect(constant_item.type_expr != null);
                }
            },
            .LogDecl => |lowered_log| {
                found_log = true;
                try testing.expectEqual(@as(usize, 3), lowered_log.fields.len);
                try testing.expect(lowered_log.fields[0].indexed);
                try testing.expect(!lowered_log.fields[1].indexed);
            },
            .ErrorDecl => |lowered_error| {
                found_error = true;
                try testing.expectEqual(@as(usize, 1), lowered_error.parameters.len);
            },
            else => {},
        }
    }

    try testing.expect(found_import);
    try testing.expect(found_field);
    try testing.expect(found_constant);
    try testing.expect(found_log);
    try testing.expect(found_error);
}

test "compiler parses tuple dot access as index expressions" {
    const source_text =
        \\contract TupleDot {
        \\    pub fn run() -> u256 {
        \\        let t: (u256, bool) = (100, true);
        \\        if (t.1) {
        \\            return t.0;
        \\        }
        \\        return 0;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    const file = try compilation.db.astFile(module.file_id);
    const contract = file.item(file.root_items[0]).Contract;
    const run_fn = file.item(contract.members[0]).Function;
    const body = file.body(run_fn.body).*;
    const contract_node = nthChildNodeOfKind(root, .ContractItem, 0).?;
    const function = nthChildNodeOfKind(contract_node, .FunctionItem, 0).?;
    const syntax_body = nthChildNodeOfKind(function, .Body, 0).?;
    try testing.expect(containsNodeOfKind(syntax_body, .IndexExpr));

    const if_stmt = file.statement(body.statements[1]).If;
    const if_condition = file.expression(if_stmt.condition).*;
    try testing.expect(if_condition == .Index);

    const then_body = file.body(if_stmt.then_body).*;
    const ret_stmt = file.statement(then_body.statements[0]).Return;
    try testing.expect(ret_stmt.value != null);
    try testing.expect(file.expression(ret_stmt.value.?).* == .Index);
}

test "compiler contextualizes typed tuple literals with mixed element types" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\pub fn sender_and_amount() -> u256 {
        \\    let t: (address, u256) = (std.msg.sender(), 500);
        \\    return t.1;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.tuple<!ora.address, i256>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
}

test "compiler contextualizes typed tuple literals with narrow integer elements" {
    const source_text =
        \\pub fn amount_only() -> u256 {
        \\    let t: (u8, u256) = (255, 1000);
        \\    return t.1;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.tuple<i8, i256>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.trunci"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
}

test "compiler supports comptime while statements" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    var total: u256 = 0;
        \\    comptime while (total < 5) {
        \\        total = total + 1;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
}

test "compiler semantic queries index names and infer expression types" {
    const source_text =
        \\pub fn add(x: u256, y: u256) -> u256 {
        \\    let z = x + y;
        \\    return z;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(item_index.lookup("add") != null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = item_index.lookup("add").? });

    var saw_integer = false;
    for (typecheck.expr_types) |expr_type| {
        if (expr_type.kind() == .integer) {
            saw_integer = true;
            break;
        }
    }
    try testing.expect(saw_integer);
}

test "compiler extracts verification facts and lowers HIR handles" {
    const source_text =
        \\contract Counter {
        \\    invariant total >= 0;
        \\    storage var total: u256;
        \\
        \\    pub fn set(next: u256) -> u256
        \\        requires next >= 0;
        \\        ensures result >= 0;
        \\    {
        \\        total = next;
        \\        return total;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const contract_id = ast_file.root_items[0];
    const facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    try testing.expect(facts.facts.len >= 1);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.items.len >= 2);
    try testing.expect(hir_result.module.raw_module.ptr != null);

    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.contract"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @set"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.global"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.requires"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
}

test "compiler HIR output runs through Z3 verification" {
    const source_text =
        \\pub fn keep(next: u256) -> u256
        \\    requires next >= 0;
        \\    ensures result >= 0;
        \\{
        \\    return next;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    const z3_verification = @import("ora_z3_verification");
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.parallel = false;

    var result = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "compiler lowers imports as namespace metadata only" {
    const source_text =
        \\comptime const std = @import("std");
        \\contract ArithmeticProbe {
        \\    pub fn add_case(a: u256, b: u256) {
        \\        let c: u256 = a + b;
        \\        assert(c >= a, "add_monotonic");
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.contract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.import"));
}

test "compiler aggregates sema and verification across multiple root items" {
    const source_text =
        \\pub fn first(x: u256) -> u256
        \\    requires x >= 0;
        \\{
        \\    return x;
        \\}
        \\
        \\pub fn second(flag: bool) -> bool
        \\    ensures result == flag;
        \\{
        \\    return flag;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 2), ast_file.root_items.len);

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const second_fn = ast_file.item(ast_file.root_items[1]).Function;
    const second_body = ast_file.body(second_fn.body);
    const second_return = ast_file.statement(second_body.statements[0]).Return;
    try testing.expectEqual(compiler.sema.TypeKind.bool, module_typecheck.exprType(second_return.value.?).kind());

    const module_facts = try compilation.db.moduleVerificationFacts(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 2), module_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, module_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, module_facts.facts[1].kind);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @first"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @second"));
}

test "compiler parses control flow and verification statements" {
    const source_text =
        \\pub fn run(values: u256) -> u256 {
        \\    for (values) |value, index|
        \\        invariant value >= index;
        \\    {
        \\        assert(value >= index, "ordered");
        \\        assume(value >= 0);
        \\        havoc state;
        \\    }
        \\
        \\    switch (values) {
        \\        0 => 1;
        \\        1...2 => {
        \\            return 2;
        \\        }
        \\        else => {
        \\            return 3;
        \\        }
        \\    }
        \\
        \\    try {
        \\        return 0;
        \\    } catch (err) {
        \\        return 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);

    try testing.expectEqual(@as(usize, 3), body.statements.len);
    try testing.expect(ast_file.statement(body.statements[0]).* == .For);
    try testing.expect(ast_file.statement(body.statements[1]).* == .Switch);
    try testing.expect(ast_file.statement(body.statements[2]).* == .Try);

    const for_stmt = ast_file.statement(body.statements[0]).For;
    try testing.expectEqual(@as(usize, 1), for_stmt.invariants.len);
    const for_body = ast_file.body(for_stmt.body);
    try testing.expectEqual(@as(usize, 3), for_body.statements.len);
    try testing.expect(ast_file.statement(for_body.statements[0]).* == .Assert);
    try testing.expect(ast_file.statement(for_body.statements[1]).* == .Assume);
    try testing.expect(ast_file.statement(for_body.statements[2]).* == .Havoc);

    const switch_stmt = ast_file.statement(body.statements[1]).Switch;
    try testing.expectEqual(@as(usize, 2), switch_stmt.arms.len);
    try testing.expect(switch_stmt.else_body != null);

    const try_stmt = ast_file.statement(body.statements[2]).Try;
    try testing.expect(try_stmt.catch_clause != null);
    try testing.expect(try_stmt.catch_clause.?.error_pattern != null);

    const resolution = try compilation.db.resolveNames(compilation.root_module_id);
    const invariant_expr = ast_file.expression(for_stmt.invariants[0]).Binary;
    try testing.expect(resolution.expr_bindings[invariant_expr.lhs.index()] != null);
    try testing.expect(resolution.expr_bindings[invariant_expr.rhs.index()] != null);
    const assert_stmt = ast_file.statement(for_body.statements[0]).Assert;
    const assert_expr = ast_file.expression(assert_stmt.condition).Binary;
    try testing.expect(resolution.expr_bindings[assert_expr.lhs.index()] != null);
    try testing.expect(resolution.expr_bindings[assert_expr.rhs.index()] != null);
}

test "compiler memoizes semantic queries per key" {
    const source_text =
        \\contract Counter {
        \\    invariant total >= 0;
        \\    storage var total: u256;
        \\
        \\    pub fn set(next: u256) -> u256
        \\        requires next >= 0;
        \\        ensures result >= 0;
        \\    {
        \\        assert(next >= 0, "non-negative");
        \\        return next;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const contract_id = ast_file.root_items[0];
    var function_id: ?compiler.ast.ItemId = null;
    for (ast_file.item(contract_id).Contract.members) |member_id| {
        if (ast_file.item(member_id).* == .Function) {
            function_id = member_id;
            break;
        }
    }

    try testing.expect(function_id != null);
    const function = ast_file.item(function_id.?).Function;

    const item_typecheck_1 = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = function_id.? });
    const item_typecheck_2 = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = function_id.? });
    try testing.expectEqual(@intFromPtr(item_typecheck_1), @intFromPtr(item_typecheck_2));

    const body_typecheck_1 = try compilation.db.typeCheck(compilation.root_module_id, .{ .body = function.body });
    const body_typecheck_2 = try compilation.db.typeCheck(compilation.root_module_id, .{ .body = function.body });
    try testing.expectEqual(@intFromPtr(body_typecheck_1), @intFromPtr(body_typecheck_2));
    try testing.expect(@intFromPtr(item_typecheck_1) != @intFromPtr(body_typecheck_1));

    const contract_facts_1 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    const contract_facts_2 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    try testing.expectEqual(@intFromPtr(contract_facts_1), @intFromPtr(contract_facts_2));

    const function_facts_1 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = function_id.? });
    const function_facts_2 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = function_id.? });
    try testing.expectEqual(@intFromPtr(function_facts_1), @intFromPtr(function_facts_2));
    try testing.expect(@intFromPtr(contract_facts_1) != @intFromPtr(function_facts_1));
    try testing.expectEqual(@as(usize, 1), contract_facts_1.facts.len);
    try testing.expectEqual(@as(usize, 2), function_facts_1.facts.len);
}

test "compiler verification facts respect item and body keys" {
    const source_text =
        \\contract Counter {
        \\    invariant total >= 0;
        \\    storage var total: u256;
        \\
        \\    pub fn set(next: u256) -> u256
        \\        requires next >= 0;
        \\        ensures result >= 0;
        \\    {
        \\        return next;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract_id = ast_file.root_items[0];
    const contract = ast_file.item(contract_id).Contract;

    var function_id: ?compiler.ast.ItemId = null;
    for (contract.members) |member_id| {
        if (ast_file.item(member_id).* == .Function) {
            function_id = member_id;
            break;
        }
    }
    try testing.expect(function_id != null);

    const function = ast_file.item(function_id.?).Function;
    const contract_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    const function_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = function_id.? });
    const body_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .body = function.body });

    try testing.expectEqual(@as(usize, 1), contract_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.invariant, contract_facts.facts[0].kind);

    try testing.expectEqual(@as(usize, 2), function_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, function_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, function_facts.facts[1].kind);

    try testing.expectEqual(@as(usize, 2), body_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, body_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, body_facts.facts[1].kind);
}

test "compiler invalidates cached queries after source update" {
    const original_source =
        \\comptime const dep = @import("old_dep");
        \\
        \\pub fn old_name() -> u256 {
        \\    return 1;
        \\}
    ;
    const updated_source =
        \\comptime const dep = @import("new_dep");
        \\
        \\pub fn new_name() -> u256 {
        \\    return 2;
        \\}
    ;

    var compilation = try compileText(original_source);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const file_id = module.file_id;

    const tree_before = try compilation.db.syntaxTree(file_id);
    const rebuilt_before = try tree_before.reconstructSource(testing.allocator);
    defer testing.allocator.free(rebuilt_before);
    try testing.expectEqualStrings(original_source, rebuilt_before);

    const graph_before = try compilation.db.moduleGraph(compilation.package_id);
    try testing.expectEqual(@as(usize, 1), graph_before.modules.len);
    try testing.expectEqual(@as(usize, 1), graph_before.modules[0].imports.len);
    try testing.expectEqualStrings("old_dep", graph_before.modules[0].imports[0].path);

    const index_before = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(index_before.lookup("old_name") != null);

    try compilation.db.updateSourceFile(file_id, updated_source);

    const tree_after = try compilation.db.syntaxTree(file_id);
    const rebuilt_after = try tree_after.reconstructSource(testing.allocator);
    defer testing.allocator.free(rebuilt_after);
    try testing.expectEqualStrings(updated_source, rebuilt_after);

    const graph_after = try compilation.db.moduleGraph(compilation.package_id);
    try testing.expectEqual(@as(usize, 1), graph_after.modules.len);
    try testing.expectEqual(@as(usize, 1), graph_after.modules[0].imports.len);
    try testing.expectEqualStrings("new_dep", graph_after.modules[0].imports[0].path);

    const index_after = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(index_after.lookup("old_name") == null);
    const new_function_id = index_after.lookup("new_name");
    try testing.expect(new_function_id != null);

    const ast_file = try compilation.db.astFile(file_id);
    try testing.expectEqual(@as(usize, 2), ast_file.root_items.len);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = new_function_id.? });

    var saw_integer = false;
    for (typecheck.expr_types) |expr_type| {
        if (expr_type.kind() == .integer) {
            saw_integer = true;
            break;
        }
    }
    try testing.expect(saw_integer);
}

test "compiler handles empty modules in module-level queries" {
    var compilation = try compileText("");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 0), ast_file.root_items.len);
    try testing.expectEqual(@as(usize, 0), ast_file.bodies.len);

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), module_typecheck.item_types.len);
    try testing.expectEqual(@as(usize, 0), module_typecheck.pattern_types.len);
    try testing.expectEqual(@as(usize, 0), module_typecheck.expr_types.len);
    try testing.expectEqual(@as(usize, 0), module_typecheck.body_types.len);
    try testing.expect(module_typecheck.diagnostics.isEmpty());

    const module_facts = try compilation.db.moduleVerificationFacts(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), module_facts.facts.len);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), hir_result.items.len);
}

test "compiler module graph resolves dependencies and detects cycles" {
    var compiler_db = compiler.db.CompilerDb.init(testing.allocator);
    defer compiler_db.deinit();

    const package_id = try compiler_db.addPackage("main");

    const c_file = try compiler_db.addSourceFile("c.ora",
        \\pub fn c() -> u256 {
        \\    return 1;
        \\}
    );
    const b_file = try compiler_db.addSourceFile("b.ora",
        \\comptime const c_dep = @import("./c.ora");
        \\pub fn b() -> u256 {
        \\    return 1;
        \\}
    );
    const a_file = try compiler_db.addSourceFile("a.ora",
        \\comptime const b_dep = @import("./b.ora");
        \\pub fn a() -> u256 {
        \\    return 1;
        \\}
    );

    const c_module = try compiler_db.addModule(package_id, c_file, "c");
    const b_module = try compiler_db.addModule(package_id, b_file, "b");
    const a_module = try compiler_db.addModule(package_id, a_file, "a");

    const graph = try compiler_db.moduleGraph(package_id);
    try testing.expectEqual(@as(usize, 3), graph.modules.len);
    try testing.expect(!graph.has_cycles);
    try testing.expectEqual(@as(usize, 3), graph.topo_order.len);
    try testing.expectEqual(c_module, graph.topo_order[0]);
    try testing.expectEqual(b_module, graph.topo_order[1]);
    try testing.expectEqual(a_module, graph.topo_order[2]);

    const c_summary = for (graph.modules) |module_summary| {
        if (module_summary.module_id == c_module) break module_summary;
    } else unreachable;
    const b_summary = for (graph.modules) |module_summary| {
        if (module_summary.module_id == b_module) break module_summary;
    } else unreachable;
    const a_summary = for (graph.modules) |module_summary| {
        if (module_summary.module_id == a_module) break module_summary;
    } else unreachable;

    try testing.expectEqual(@as(usize, 0), c_summary.dependencies.len);

    try testing.expectEqual(@as(usize, 1), b_summary.dependencies.len);
    try testing.expectEqual(c_module, b_summary.dependencies[0]);
    try testing.expectEqual(c_module, b_summary.imports[0].target_module_id.?);

    try testing.expectEqual(@as(usize, 1), a_summary.dependencies.len);
    try testing.expectEqual(b_module, a_summary.dependencies[0]);
    try testing.expectEqual(b_module, a_summary.imports[0].target_module_id.?);

    try compiler_db.updateSourceFile(c_file,
        \\comptime const a_dep = @import("./a.ora");
        \\pub fn c() -> u256 {
        \\    return 1;
        \\}
    );

    const cycle_graph = try compiler_db.moduleGraph(package_id);
    try testing.expect(cycle_graph.has_cycles);
    const cycle_c_summary = for (cycle_graph.modules) |module_summary| {
        if (module_summary.module_id == c_module) break module_summary;
    } else unreachable;
    try testing.expectEqual(@as(usize, 1), cycle_c_summary.dependencies.len);
    try testing.expectEqual(a_module, cycle_c_summary.dependencies[0]);
}

test "compiler invalidates dependent module caches after source update" {
    var compiler_db = compiler.db.CompilerDb.init(testing.allocator);
    defer compiler_db.deinit();

    const package_id = try compiler_db.addPackage("main");

    const c_file = try compiler_db.addSourceFile("c.ora",
        \\pub fn c() -> u256 {
        \\    return 1;
        \\}
    );
    const b_file = try compiler_db.addSourceFile("b.ora",
        \\comptime const c_dep = @import("./c.ora");
        \\pub fn b() -> u256 {
        \\    return 2;
        \\}
    );
    const a_file = try compiler_db.addSourceFile("a.ora",
        \\comptime const b_dep = @import("./b.ora");
        \\pub fn a() -> u256 {
        \\    return 3;
        \\}
    );

    const c_module = try compiler_db.addModule(package_id, c_file, "c");
    const b_module = try compiler_db.addModule(package_id, b_file, "b");
    const a_module = try compiler_db.addModule(package_id, a_file, "a");

    _ = try compiler_db.moduleGraph(package_id);

    const b_index = try compiler_db.itemIndex(b_module);
    const a_index = try compiler_db.itemIndex(a_module);
    const b_item = b_index.lookup("b").?;
    const a_item = a_index.lookup("a").?;

    const b_typecheck_before = try compiler_db.typeCheck(b_module, .{ .item = b_item });
    const a_typecheck_before = try compiler_db.typeCheck(a_module, .{ .item = a_item });
    const a_hir_before = try compiler_db.lowerToHir(a_module);

    try compiler_db.updateSourceFile(c_file,
        \\pub fn c() -> u256 {
        \\    return 99;
        \\}
    );

    const b_typecheck_after = try compiler_db.typeCheck(b_module, .{ .item = b_item });
    const a_typecheck_after = try compiler_db.typeCheck(a_module, .{ .item = a_item });
    const a_hir_after = try compiler_db.lowerToHir(a_module);
    const graph_after = try compiler_db.moduleGraph(package_id);

    try testing.expect(b_typecheck_before != b_typecheck_after);
    try testing.expect(a_typecheck_before != a_typecheck_after);
    try testing.expect(a_hir_before != a_hir_after);
    try testing.expectEqual(@as(usize, 3), graph_after.modules.len);

    _ = c_module;
}

test "compiler package loader bridges import graph into source modules" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
        \\pub fn helper() -> u256 {
        \\    return 7;
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\pub fn run() -> u256 {
        \\    return 1;
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const package = compilation.db.sources.package(compilation.package_id);
    try testing.expectEqual(@as(usize, 2), package.modules.items.len);
    try testing.expectEqualStrings("main", compilation.db.sources.module(compilation.root_module_id).name);

    const graph = try compilation.db.moduleGraph(compilation.package_id);
    try testing.expectEqual(@as(usize, 2), graph.modules.len);

    const root_summary = for (graph.modules) |summary| {
        if (summary.module_id == compilation.root_module_id) break summary;
    } else unreachable;
    try testing.expectEqual(@as(usize, 1), root_summary.imports.len);
    try testing.expect(root_summary.imports[0].target_module_id != null);
}

test "compiler module graph distinguishes imports with the same basename in different directories" {
    var compiler_db = compiler.db.CompilerDb.init(testing.allocator);
    defer compiler_db.deinit();

    const package_id = try compiler_db.addPackage("main");

    const left_math_file = try compiler_db.addSourceFile("left/math.ora",
        \\pub fn left() -> u256 { return 1; }
    );
    const right_math_file = try compiler_db.addSourceFile("right/math.ora",
        \\pub fn right() -> u256 { return 2; }
    );
    const main_file = try compiler_db.addSourceFile("main.ora",
        \\comptime const left_math = @import("./left/math.ora");
        \\comptime const right_math = @import("./right/math.ora");
        \\pub fn run() -> u256 { return 0; }
    );

    const left_math_module = try compiler_db.addModule(package_id, left_math_file, "math");
    const right_math_module = try compiler_db.addModule(package_id, right_math_file, "math");
    const main_module = try compiler_db.addModule(package_id, main_file, "main");

    const graph = try compiler_db.moduleGraph(package_id);
    const main_summary = for (graph.modules) |summary| {
        if (summary.module_id == main_module) break summary;
    } else unreachable;

    try testing.expectEqual(@as(usize, 2), main_summary.imports.len);
    try testing.expectEqual(left_math_module, main_summary.imports[0].target_module_id.?);
    try testing.expectEqual(right_math_module, main_summary.imports[1].target_module_id.?);
    try testing.expect(std.mem.indexOfScalar(compiler.source.ModuleId, main_summary.dependencies, left_math_module) != null);
    try testing.expect(std.mem.indexOfScalar(compiler.source.ModuleId, main_summary.dependencies, right_math_module) != null);
}

test "compiler const eval executes cross-module comptime calls" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
        \\comptime fn helper() -> u256 {
        \\    return 7;
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        dep.helper();
        \\    };
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 7), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler parses log, error, and bitfield declarations" {
    const source_text =
        \\contract Ledger {
        \\    bitfield Flags: u256 {
        \\        enabled: bool(1);
        \\        mode: u8 @bits(1..9);
        \\    }
        \\
        \\    log Transfer(indexed from: address, to: address);
        \\    error Failure(code: u256);
        \\    storage var total: u256;
        \\
        \\    pub fn emit_transfer(to: address) -> u256 {
        \\        log Transfer(total, to);
        \\        return total;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const ast_diags = try compilation.db.astDiagnostics(compilation.db.sources.module(compilation.root_module_id).file_id);
    try testing.expect(ast_diags.isEmpty());
    const contract = ast_file.item(ast_file.root_items[0]).Contract;

    try testing.expectEqual(@as(usize, 5), contract.members.len);
    try testing.expect(ast_file.item(contract.members[0]).* == .Bitfield);
    try testing.expect(ast_file.item(contract.members[1]).* == .LogDecl);
    try testing.expect(ast_file.item(contract.members[2]).* == .ErrorDecl);
    try testing.expect(ast_file.item(contract.members[3]).* == .Field);
    try testing.expect(ast_file.item(contract.members[4]).* == .Function);

    const bitfield = ast_file.item(contract.members[0]).Bitfield;
    try testing.expectEqualStrings("Flags", bitfield.name);
    try testing.expect(bitfield.base_type != null);
    try testing.expectEqual(@as(usize, 2), bitfield.fields.len);
    try testing.expect(bitfield.fields[0].width != null);
    try testing.expectEqual(@as(u32, 1), bitfield.fields[0].width.?);
    try testing.expect(bitfield.fields[1].offset != null);
    try testing.expect(bitfield.fields[1].width != null);
    try testing.expectEqual(@as(u32, 1), bitfield.fields[1].offset.?);
    try testing.expectEqual(@as(u32, 8), bitfield.fields[1].width.?);

    const log_decl = ast_file.item(contract.members[1]).LogDecl;
    try testing.expectEqualStrings("Transfer", log_decl.name);
    try testing.expectEqual(@as(usize, 2), log_decl.fields.len);
    try testing.expect(log_decl.fields[0].indexed);
    try testing.expect(!log_decl.fields[1].indexed);

    const error_decl = ast_file.item(contract.members[2]).ErrorDecl;
    try testing.expectEqualStrings("Failure", error_decl.name);
    try testing.expectEqual(@as(usize, 1), error_decl.parameters.len);

    const function = ast_file.item(contract.members[4]).Function;
    const body = ast_file.body(function.body);
    try testing.expectEqual(@as(usize, 2), body.statements.len);
    try testing.expect(ast_file.statement(body.statements[0]).* == .Log);

    const log_stmt = ast_file.statement(body.statements[0]).Log;
    try testing.expectEqualStrings("Transfer", log_stmt.name);
    try testing.expectEqual(@as(usize, 2), log_stmt.args.len);

    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(item_index.lookup("Flags") != null);
    try testing.expect(item_index.lookup("Transfer") != null);
    try testing.expect(item_index.lookup("Failure") != null);
    try testing.expect(item_index.lookup("Ledger.Transfer") != null);

    const resolution = try compilation.db.resolveNames(compilation.root_module_id);
    try testing.expect(resolution.expr_bindings[log_stmt.args[0].index()] != null);
    try testing.expect(resolution.expr_bindings[log_stmt.args[1].index()] != null);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var saw_bitfield = false;
    var saw_log_decl = false;
    var saw_error_decl = false;
    for (hir_result.items) |item| {
        switch (item.kind) {
            .bitfield => saw_bitfield = true,
            .log_decl => saw_log_decl = true,
            .error_decl => saw_error_decl = true,
            else => {},
        }
    }
    try testing.expect(saw_bitfield);
    try testing.expect(saw_log_decl);
    try testing.expect(saw_error_decl);

    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.log_decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error_decl"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.log_decl\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.error_decl\""));
}

test "compiler lowers structured type expressions" {
    const source_text =
        \\struct Types {
        \\    values: [u256; 5];
        \\    view: slice[address];
        \\    table: map<address, map<address, u256>>;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn use_types(
        \\    buffer: slice[u256],
        \\    pair: (u8, bool),
        \\    bounded: MinValue<u256, 100>,
        \\    maybe: !u256 | Failure,
        \\) -> slice[u256] {
        \\    return buffer;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const ast_diags = try compilation.db.astDiagnostics(compilation.db.sources.module(compilation.root_module_id).file_id);
    try testing.expect(ast_diags.isEmpty());

    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;
    try testing.expect(ast_file.typeExpr(struct_item.fields[0].type_expr).* == .Array);
    try testing.expect(ast_file.typeExpr(struct_item.fields[1].type_expr).* == .Slice);
    try testing.expect(ast_file.typeExpr(struct_item.fields[2].type_expr).* == .Generic);

    const array_type = ast_file.typeExpr(struct_item.fields[0].type_expr).Array;
    try testing.expect(array_type.size == .Integer);
    try testing.expectEqualStrings("5", array_type.size.Integer.text);
    try testing.expect(ast_file.typeExpr(array_type.element).* == .Path);
    try testing.expectEqualStrings("u256", ast_file.typeExpr(array_type.element).Path.name);

    const map_type = ast_file.typeExpr(struct_item.fields[2].type_expr).Generic;
    try testing.expectEqualStrings("map", map_type.name);
    try testing.expectEqual(@as(usize, 2), map_type.args.len);
    try testing.expect(map_type.args[0] == .Type);
    try testing.expect(map_type.args[1] == .Type);
    try testing.expect(ast_file.typeExpr(map_type.args[1].Type).* == .Generic);

    const function = ast_file.item(ast_file.root_items[2]).Function;
    try testing.expectEqual(@as(usize, 4), function.parameters.len);
    try testing.expect(ast_file.typeExpr(function.parameters[0].type_expr).* == .Slice);
    try testing.expect(ast_file.typeExpr(function.parameters[1].type_expr).* == .Tuple);
    try testing.expect(ast_file.typeExpr(function.parameters[2].type_expr).* == .Generic);
    try testing.expect(ast_file.typeExpr(function.parameters[3].type_expr).* == .ErrorUnion);
    try testing.expect(ast_file.typeExpr(function.return_type.?).* == .Slice);

    const bounded_type = ast_file.typeExpr(function.parameters[2].type_expr).Generic;
    try testing.expectEqualStrings("MinValue", bounded_type.name);
    try testing.expectEqual(@as(usize, 2), bounded_type.args.len);
    try testing.expect(bounded_type.args[0] == .Type);
    try testing.expect(bounded_type.args[1] == .Integer);
    try testing.expectEqualStrings("100", bounded_type.args[1].Integer.text);

    const maybe_type = ast_file.typeExpr(function.parameters[3].type_expr).ErrorUnion;
    try testing.expect(ast_file.typeExpr(maybe_type.payload).* == .Path);
    try testing.expectEqual(@as(usize, 1), maybe_type.errors.len);
    try testing.expectEqualStrings("Failure", ast_file.typeExpr(maybe_type.errors[0]).Path.name);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    const bounded_param = function.parameters[2];
    const bounded_located_type = typecheck.pattern_types[bounded_param.pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.refinement, bounded_located_type.kind());
    try testing.expectEqualStrings("MinValue", bounded_located_type.name().?);
    try testing.expect(bounded_located_type.type.refinementBaseType() != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, bounded_located_type.type.refinementBaseType().?.kind());

    const body = ast_file.body(function.body);
    const return_stmt = ast_file.statement(body.statements[0]).Return;
    const return_expr = return_stmt.value.?;
    try testing.expectEqual(compiler.sema.TypeKind.slice, typecheck.exprType(return_expr).kind());
    try testing.expect(typecheck.exprType(return_expr).elementType() != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(return_expr).elementType().?.kind());

    const buffer_type = typecheck.pattern_types[function.parameters[0].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.slice, buffer_type.kind());
    try testing.expect(buffer_type.elementType() != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, buffer_type.elementType().?.kind());

    const maybe_typecheck = typecheck.pattern_types[function.parameters[3].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.error_union, maybe_typecheck.kind());
    try testing.expect(maybe_typecheck.payloadType() != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, maybe_typecheck.payloadType().?.kind());
    try testing.expectEqual(@as(usize, 1), maybe_typecheck.errorTypes().len);
    try testing.expectEqual(compiler.sema.TypeKind.named, maybe_typecheck.errorTypes()[0].kind());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.min_value"));
}

test "compiler inserts parameter refinement guards in HIR" {
    const source_text =
        \\pub fn guarded(
        \\    bounded: MinValue<u256, 100>,
        \\    target: NonZeroAddress,
        \\) -> u256 {
        \\    return bounded;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "parameter_refinement"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "MinValue"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_to_base"));
}

test "compiler inserts refinement flow conversions in HIR" {
    const source_text =
        \\pub fn promote(value: u256) -> MinValue<u256, 100> {
        \\    return @cast(MinValue<u256, 100>, value);
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.base_to_refinement"));
}

test "compiler inserts refinement call conversions in HIR" {
    const source_text =
        \\fn take_base(value: u256) -> u256 {
        \\    return value;
        \\}
        \\
        \\fn take_refined(value: MinValue<u256, 100>) -> MinValue<u256, 100> {
        \\    return value;
        \\}
        \\
        \\pub fn bridge(
        \\    raw: u256,
        \\    bounded: MinValue<u256, 100>,
        \\) -> u256 {
        \\    let left = take_base(bounded);
        \\    let right = take_refined(raw);
        \\    return left;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.refinement_to_base"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.base_to_refinement"));
}

test "compiler cleans refinement guards to cf.assert" {
    const source_text =
        \\pub fn guarded(
        \\    bounded: MinValue<u256, 100>,
        \\) -> u256 {
        \\    return bounded;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const mutable_hir_result = @constCast(hir_result);
    var empty_guards = std.StringHashMap(void).init(testing.allocator);
    defer empty_guards.deinit();

    mutable_hir_result.cleanupRefinementGuards(&empty_guards);
    const hir_text = try mutable_hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "cf.assert"));
}

test "compiler lowers struct and enum declarations through real decl ops" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: bool;
        \\}
        \\
        \\enum Mode {
        \\    off,
        \\    on,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct.decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum.decl"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.struct_decl\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.enum_decl\""));
}

test "compiler lowers struct instantiate and field extract through real ops" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn read() -> u256 {
        \\    let pair = Pair { right: 2, left: 1 };
        \\    return pair.left;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.struct.create\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers enum variant access through real enum constant op" {
    const source_text =
        \\enum Mode {
        \\    off,
        \\    on,
        \\}
        \\
        \\pub fn current() -> Mode {
        \\    return Mode.on;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.constant"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers generic named struct field access without placeholder" {
    const source_text =
        \\struct Book(T) {
        \\    sender_before: T;
        \\    amount: T;
        \\}
        \\
        \\pub fn read() -> u256 {
        \\    let book = Book(u256) { sender_before: 1, amount: 2 };
        \\    return book.sender_before;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers shorthand storage field assignments" {
    const source_text =
        \\contract Wallet {
        \\    storage owner: address;
        \\
        \\    pub fn setOwner(next: address) {
        \\        owner = next;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
}

test "compiler preserves typed local names in assignments" {
    const source_text =
        \\contract Wallet {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn deposit(amount: u256) {
        \\        const sender: address = std.msg.sender;
        \\        balances[sender] = amount;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_store"));
}

test "compiler wraps payload returns into real error ok op" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn lift(value: u256) -> !u256 | Failure {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.ok"));
}

test "compiler lowers try expressions through real error helper ops" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn lift(maybe: !u256 | Failure) -> !u256 | Failure {
        \\    return try maybe;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.is_error"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.get_error"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.err"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.unwrap"));
}

test "compiler carries tuple-payload error unions through HIR" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn probe(maybe: !(u256, bool) | Failure) -> !bool | Failure {
        \\    try maybe;
        \\    return true;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.error_union<!ora.tuple<i256, i1>>"));
}

test "compiler marks payload-bearing narrow error unions for wide lowering in HIR" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn helper(flag: bool) -> !bool | Failure {
        \\    return flag;
        \\}
        \\
        \\pub fn use(flag: bool) -> !bool | Failure {
        \\    return try helper(flag);
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.force_wide_error_union"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.ok"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.err"));
}

test "compiler lowers struct field mutation through real field update op" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn update() -> u256 {
        \\    let pair = Pair { left: 1, right: 2 };
        \\    pair.left = 42;
        \\    return pair.left;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_update"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers string, bytes, and address literals through real ops" {
    const source_text =
        \\pub fn owner() -> address {
        \\    let note = "hello";
        \\    let payload = hex"deadbeef";
        \\    let who = 0x1234567890abcdef1234567890abcdef12345678;
        \\    return who;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.string.constant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bytes.constant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.i160.to.addr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.string_const"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.bytes_const"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.address_const"));
}

test "compiler does not lower type-value expression statements to placeholder ops" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        Pair<u256>;
        \\        1;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.type_value"));
}

test "compiler lowers top-level const items through ora.const" {
    const source_text =
        \\const LIMIT: u256 = 2;
        \\const READY: bool = true;
        \\
        \\pub fn run() -> u256 {
        \\    if (READY) {
        \\        return LIMIT;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.const"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "LIMIT"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "READY"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.constant_decl"));
}

test "compiler lowers literal top-level const items through ora.const" {
    const source_text =
        \\const NAME: string = "ora";
        \\const PAYLOAD: bytes = hex"deadbeef";
        \\const OWNER: address = 0x1234567890abcdef1234567890abcdef12345678;
        \\
        \\pub fn run() -> u256 {
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 3, "ora.const"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "NAME"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "PAYLOAD"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "OWNER"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"ora\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"deadbeef\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.constant_decl"));
}

test "compiler lowers immutable contract fields through ora.immutable" {
    const source_text =
        \\contract C {
        \\    immutable OWNER: address = 0x1234567890abcdef1234567890abcdef12345678;
        \\
        \\    pub fn owner() -> address {
        \\        return OWNER;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.immutable"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "OWNER"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.i160.to.addr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.field_decl"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.immutable_decl"));
}

test "compiler lowers computed top-level const items through ora.const" {
    const source_text =
        \\const LIMIT: u256 = 1 + 2 * 3;
        \\const READY: bool = 4 > 3;
        \\
        \\pub fn run() -> u256 {
        \\    if (READY) {
        \\        return LIMIT;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.const"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "LIMIT"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "READY"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.constant_decl"));
}

test "compiler lowers bitfield types as wire integers with metadata attrs" {
    const source_text =
        \\contract Bits {
        \\    bitfield Flags: u256 {
        \\        enabled: u1,
        \\        signed: i8,
        \\    }
        \\
        \\    storage packed: Flags;
        \\
        \\    pub fn use_bits(flag: Flags) -> u256 {
        \\        packed = flag;
        \\        return packed;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.bitfield = \"Flags\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bitfield_layout"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "i256"));
}

test "compiler tracks per-function read and write effects" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\    tstore var pending: u256;
        \\
        \\    pub fn read_only() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    pub fn write_only(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn mixed(value: u256) -> u256 {
        \\        pending += value;
        \\        return total;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);

    const read_only = item_index.lookup("read_only").?;
    const write_only = item_index.lookup("write_only").?;
    const mixed = item_index.lookup("mixed").?;

    switch (typecheck.itemEffect(read_only)) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(write_only)) {
        .writes => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(mixed)) {
        .reads_writes => |effect| {
            try testing.expect(containsEffectSlot(effect.reads, "pending", .transient));
            try testing.expect(containsEffectSlot(effect.reads, "total", .storage));
            try testing.expect(containsEffectSlot(effect.writes, "pending", .transient));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler composes callee effects into caller summaries" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\    tstore var pending: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    fn write_pending(value: u256) {
        \\        pending = value;
        \\    }
        \\
        \\    pub fn wrapper(value: u256) -> u256 {
        \\        write_pending(value);
        \\        return (read_total());
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const wrapper = item_index.lookup("wrapper").?;

    switch (typecheck.itemEffect(wrapper)) {
        .reads_writes => |effect| {
            try testing.expect(containsEffectSlot(effect.reads, "total", .storage));
            try testing.expect(containsEffectSlot(effect.writes, "pending", .transient));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks keyed map effects by parameter" {
    const source_text =
        \\contract Effects {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn read_balance(user: address) -> u256 {
        \\        return balances[user];
        \\    }
        \\
        \\    pub fn write_balance(user: address, value: u256) {
        \\        balances[user] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const read_balance = item_index.lookup("read_balance").?;
    const write_balance = item_index.lookup("write_balance").?;
    const user_key = [_]compiler.sema.KeySegment{.{ .parameter = 0 }};

    switch (typecheck.itemEffect(read_balance)) {
        .reads => |effect| try testing.expect(containsKeyedEffectSlot(effect.slots, "balances", .storage, &user_key)),
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(write_balance)) {
        .reads_writes => |effect| {
            try testing.expect(containsKeyedEffectSlot(effect.reads, "balances", .storage, &user_key));
            try testing.expect(containsKeyedEffectSlot(effect.writes, "balances", .storage, &user_key));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks per-expression keyed index effects" {
    const source_text =
        \\contract Effects {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn read_balance(user: address) -> u256 {
        \\        return balances[user];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const function = ast_file.item(contract.members[1]).Function;
    const ret_stmt = ast_file.statement(ast_file.body(function.body).statements[0]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const user_key = [_]compiler.sema.KeySegment{.{ .parameter = 0 }};

    switch (typecheck.exprEffect(ret_stmt.value.?)) {
        .reads => |effect| try testing.expect(containsKeyedEffectSlot(effect.slots, "balances", .storage, &user_key)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler rejects direct writes to locked slots" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn write_while_locked(value: u256) {
        \\        @lock(total);
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler rejects keyed writes to the same locked map entry" {
    const source_text =
        \\contract Locked {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn guarded(user: address, value: u256) {
        \\        @lock(balances[user]);
        \\        balances[user] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'balances'"));
}

test "compiler allows writes to a different keyed map entry" {
    const source_text =
        \\contract Locked {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn guarded(user: address, other: address, value: u256) {
        \\        @lock(balances[user]);
        \\        balances[other] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'balances'"));
}

test "compiler allows writes to a different constant keyed map entry" {
    const source_text =
        \\contract Locked {
        \\    storage counts: map<u256, u256>;
        \\
        \\    pub fn guarded(value: u256) {
        \\        @lock(counts[1]);
        \\        counts[2] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'counts'"));
}

test "compiler rejects callee writes to locked slots" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    fn write_total(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn write_while_locked(value: u256) {
        \\        @lock(total);
        \\        write_total(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler allows writes after unlock" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn write_after_unlock(value: u256) {
        \\        @lock(total);
        \\        @unlock(total);
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler allows writes after a conditional lock on only one path" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn guarded(flag: bool, value: u256) {
        \\        if (flag) {
        \\            @lock(total);
        \\        }
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler allows writes after a conditional map-entry lock on only one path" {
    const source_text =
        \\contract Locked {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn guarded(flag: bool, user: address, value: u256) {
        \\        if (flag) {
        \\            @lock(balances[user]);
        \\        }
        \\        balances[user] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'balances'"));
}

test "compiler rejects writes when all branches keep a slot locked" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    pub fn guarded(flag: bool, value: u256) {
        \\        if (flag) {
        \\            @lock(total);
        \\        } else {
        \\            @lock(total);
        \\        }
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler rejects writes to locked transient slots" {
    const source_text =
        \\contract Locked {
        \\    tstore var pending: u256;
        \\
        \\    pub fn write_while_locked(value: u256) {
        \\        @lock(pending);
        \\        pending = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked transient slot 'pending'"));
}

test "compiler composes contract member call effects into caller summaries" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\}
        \\
        \\pub fn wrapper(vault: Vault) -> u256 {
        \\    return vault.read_total();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    switch (typecheck.itemEffect(ast_file.root_items[1])) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler rejects locked writes through contract member calls" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256;
        \\
        \\    fn write_total(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn guarded(vault: Vault, value: u256) {
        \\        @lock(total);
        \\        vault.write_total(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler composes effects through local function aliases" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    pub fn wrapper() -> u256 {
        \\        let reader = read_total;
        \\        return reader();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const wrapper = item_index.lookup("wrapper").?;

    switch (typecheck.itemEffect(wrapper)) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks per-expression composed call effects" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\
        \\    pub fn wrapper() -> u256 {
        \\        return read_total();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const wrapper = ast_file.item(contract.members[2]).Function;
    const ret_stmt = ast_file.statement(ast_file.body(wrapper.body).statements[0]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    switch (typecheck.exprEffect(ret_stmt.value.?)) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler rejects locked writes through local function aliases" {
    const source_text =
        \\contract Locked {
        \\    storage total: u256;
        \\
        \\    fn write_total(value: u256) {
        \\        total = value;
        \\    }
        \\
        \\    pub fn guarded(value: u256) {
        \\        let writer = write_total;
        \\        @lock(total);
        \\        writer(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'total'"));
}

test "compiler composes effects through member-derived function aliases" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256;
        \\
        \\    fn read_total() -> u256 {
        \\        return total;
        \\    }
        \\}
        \\
        \\pub fn wrapper(vault: Vault) -> u256 {
        \\    let reader = vault.read_total;
        \\    return reader();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    switch (typecheck.itemEffect(ast_file.root_items[1])) {
        .reads => |effect| try testing.expect(containsEffectSlot(effect.slots, "total", .storage)),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks log and havoc effect kinds" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    pub fn noisy(value: u256) {
        \\        log Transfer(value);
        \\        havoc total;
        \\    }
        \\
        \\    pub fn wrapper(value: u256) {
        \\        noisy(value);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const noisy = item_index.lookup("noisy").?;
    const wrapper = item_index.lookup("wrapper").?;

    switch (typecheck.itemEffect(noisy)) {
        .side_effects => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
        },
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(wrapper)) {
        .side_effects => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
        },
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(effect.has_havoc);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler tracks lock and unlock effect kinds" {
    const source_text =
        \\contract Locks {
        \\    storage total: u256;
        \\
        \\    pub fn guarded() {
        \\        @lock(total);
        \\        @unlock(total);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const guarded = item_index.lookup("guarded").?;

    switch (typecheck.itemEffect(guarded)) {
        .side_effects => |effect| {
            try testing.expect(effect.has_lock);
            try testing.expect(effect.has_unlock);
        },
        .reads => |effect| {
            try testing.expect(effect.has_lock);
            try testing.expect(effect.has_unlock);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_lock);
            try testing.expect(effect.has_unlock);
            try testing.expect(containsEffectSlot(effect.reads, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler composes effects to a fixpoint across mutual recursion" {
    const source_text =
        \\contract Effects {
        \\    storage total: u256;
        \\
        \\    fn ping(n: u256) {
        \\        if (n == 0) {
        \\            return;
        \\        }
        \\        log Ping(n);
        \\        pong(n - 1);
        \\    }
        \\
        \\    fn pong(n: u256) {
        \\        if (n == 0) {
        \\            return;
        \\        }
        \\        total = n;
        \\        ping(n - 1);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const ping = item_index.lookup("ping").?;
    const pong = item_index.lookup("pong").?;

    switch (typecheck.itemEffect(ping)) {
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }

    switch (typecheck.itemEffect(pong)) {
        .writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.slots, "total", .storage));
        },
        .reads_writes => |effect| {
            try testing.expect(effect.has_log);
            try testing.expect(containsEffectSlot(effect.writes, "total", .storage));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "compiler lowers bitfield field reads and writes through bit ops" {
    const source_text =
        \\contract Bits {
        \\    bitfield Flags: u256 {
        \\        enabled: u1,
        \\        signed: i8,
        \\    }
        \\
        \\    storage packed: Flags;
        \\
        \\    pub fn update(flag: Flags) -> i8 {
        \\        packed = flag;
        \\        packed.enabled = 1;
        \\        packed.signed = -2;
        \\        return packed.signed;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shrui"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shrsi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.andi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.ori"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shli"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers bitfield construction through packed bit ops" {
    const source_text =
        \\contract Bits {
        \\    bitfield Flags: u256 {
        \\        enabled: u1,
        \\        signed: i8,
        \\    }
        \\
        \\    pub fn build() -> i8 {
        \\        let packed = Flags { enabled: 1, signed: -2 };
        \\        return packed.signed;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shli"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.ori"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shrui"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.struct.create\""));
}

test "compiler resolves named path types to declaration kinds" {
    const source_text =
        \\struct Pair {
        \\    x: u256;
        \\}
        \\
        \\enum Mode: u8 {
        \\    idle,
        \\}
        \\
        \\pub fn wrap(pair: Pair, mode: Mode) -> Pair {
        \\    return pair;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    const pair_type = typecheck.pattern_types[function.parameters[0].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.struct_, pair_type.kind());

    const mode_type = typecheck.pattern_types[function.parameters[1].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.enum_, mode_type.kind());

    const return_type = typecheck.body_types[function.body.index()];
    try testing.expectEqual(compiler.sema.TypeKind.struct_, return_type.kind());
}

test "compiler tracks declaration root regions in type check output" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\    memory var scratch: u256;
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn inspect(value: u256) -> u256 {
        \\    let local = value;
        \\    return local;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    try testing.expectEqual(compiler.sema.Region.storage, typecheck.itemLocatedType(contract.members[0]).region);
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.itemLocatedType(contract.members[1]).region);
    try testing.expectEqual(compiler.sema.Region.transient, typecheck.itemLocatedType(contract.members[2]).region);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const parameter_type = typecheck.pattern_types[function.parameters[0].pattern.index()];
    try testing.expectEqual(compiler.sema.Region.memory, parameter_type.region);
    try testing.expectEqual(compiler.sema.Provenance.calldata, parameter_type.provenance);

    const body = ast_file.body(function.body);
    const local_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const local_type = typecheck.pattern_types[local_stmt.pattern.index()];
    try testing.expectEqual(compiler.sema.Region.memory, local_type.region);
    try testing.expectEqual(compiler.sema.Provenance.calldata, local_type.provenance);
}

test "compiler allows implicit region reads into locals" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn inspect(value: u256) -> u256 {
        \\    let from_param = value;
        \\    let from_storage = total;
        \\    let from_tstore = pending;
        \\    return from_param + from_storage + from_tstore;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler rejects writes to calldata and direct storage transient transfer" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn inspect(value: u256) -> u256 {
        \\    value = total;
        \\    pending = total;
        \\    total = pending;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(diags.len() >= 2);
    try testing.expect(diagnosticMessagesContain(diags, "assignment expects region 'transient'"));
    try testing.expect(diagnosticMessagesContain(diags, "assignment expects region 'storage'"));
}

test "compiler infers field and index access types" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: bool;
        \\}
        \\
        \\contract Box {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn probe(pair: Pair, values: [u256; 2], table: map<address, bool>) -> bool {
        \\    let a = pair.first;
        \\    let b = values[0];
        \\    let c = table[0x1234567890abcdef1234567890abcdef12345678];
        \\    let d = (1, false)[1];
        \\    return c;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    const a_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const b_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const c_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const d_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[4]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[a_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[b_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[c_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[d_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler infers direct and member call return types" {
    const source_text =
        \\contract Ledger {
        \\    pub fn amount() -> u256 {
        \\        return 1;
        \\    }
        \\}
        \\
        \\pub fn helper() -> bool {
        \\    return true;
        \\}
        \\
        \\pub fn probe() -> u256 {
        \\    let a = helper();
        \\    let b = Ledger.amount();
        \\    return b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    const a_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const b_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[2]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[a_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[b_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler infers operator result types from operand compatibility" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn probe(flag: bool, value: u256, other: u8, maybe: !u256 | Failure) -> u256 {
        \\    let sum = value + other;
        \\    let bits = value & other;
        \\    let cmp = value < other;
        \\    let logic = flag && true;
        \\    let negated = -value;
        \\    let failed = !value;
        \\    let unwrapped = try maybe;
        \\    return unwrapped;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    const sum_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const bits_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const cmp_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const logic_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const negated_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const failed_stmt = ast_file.statement(body.statements[5]).VariableDecl;
    const unwrapped_stmt = ast_file.statement(body.statements[6]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[7]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[sum_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[bits_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[cmp_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[logic_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[negated_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[failed_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[unwrapped_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler tracks HIR unknown type fallbacks" {
    const source_text =
        \\pub fn probe(value: u256) -> u256 {
        \\    return value.missing;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(type_diags.items.items.len > 0);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expectEqual(compiler.hir.TypeFallbackReason.sema_unknown, hir_result.type_fallbacks[0].reason);
    try testing.expectEqual(module.file_id, hir_result.type_fallbacks[0].location.file_id);
}

test "compiler lowers tuple locals without tuple fallback" {
    const source_text =
        \\pub fn probe() -> u256 {
        \\    let coords = (1, 2);
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    for (hir_result.type_fallbacks) |fallback| {
        try testing.expect(fallback.reason != .unsupported_tuple_sema_type);
        try testing.expect(fallback.reason != .unsupported_syntax_type);
    }
}

test "compiler lowers tuple HIR types without tuple fallback" {
    const source_text =
        \\pub fn pair() -> (u256, bool) {
        \\    let coords = (1, true);
        \\    return coords;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    for (hir_result.type_fallbacks) |fallback| {
        try testing.expect(fallback.reason != .unsupported_tuple_sema_type);
        try testing.expect(fallback.reason != .unsupported_syntax_type);
    }

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.tuple<i256, i1>"));
}

test "compiler lowers tuple expressions through real tuple ops" {
    const source_text =
        \\pub fn pair() -> u256 {
        \\    let coords = (1, true);
        \\    return coords[0];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.tuple.create\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.index_access\""));
}

test "compiler lowers function-valued bindings without function fallback" {
    const source_text =
        \\fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\
        \\pub fn probe() -> u256 {
        \\    let f = helper;
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    for (hir_result.type_fallbacks) |fallback| {
        try testing.expect(fallback.reason != .unsupported_function_sema_type);
    }
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.function_ref"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.function<"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.function_ref\""));
}

test "compiler syntax splits nested generic closing tokens" {
    const source_text =
        \\struct Types {
        \\    table: map<address, map<address, u256>>;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const outer_generic = nthDescendantNodeOfKind(root, .GenericType, 0).?;
    const inner_generic = nthDescendantNodeOfKind(root, .GenericType, 1).?;

    const outer_close = outer_generic.lastToken().?;
    const inner_close = inner_generic.lastToken().?;

    try testing.expectEqual(compiler.syntax.TokenKind.Greater, outer_close.kind());
    try testing.expectEqual(compiler.syntax.TokenKind.Greater, inner_close.kind());
    try testing.expect(outer_close.id != inner_close.id);
    try testing.expectEqual(@as(usize, 1), outer_close.range().end - outer_close.range().start);
    try testing.expectEqual(@as(usize, 1), inner_close.range().end - inner_close.range().start);
}

test "compiler preserves generic function template parameters in AST" {
    const source_text =
        \\contract Math {
        \\    fn max(comptime T: type, a: T, b: T) -> T {
        \\        return a;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const function = ast_file.item(contract.members[0]).Function;

    try testing.expect(function.is_generic);
    try testing.expectEqual(@as(usize, 3), function.parameters.len);
    try testing.expect(function.parameters[0].is_comptime);
    try testing.expect(ast_file.typeExpr(function.parameters[0].type_expr).* == .Path);
    try testing.expectEqualStrings("type", ast_file.typeExpr(function.parameters[0].type_expr).Path.name);
    try testing.expect(!function.parameters[1].is_comptime);
}

test "compiler preserves generic struct and contract template metadata in AST" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\contract Box(comptime T: type) {
        \\    let value: T;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);

    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;
    try testing.expect(struct_item.is_generic);
    try testing.expectEqual(@as(usize, 1), struct_item.template_parameters.len);
    try testing.expect(struct_item.template_parameters[0].is_comptime);

    const contract_item = ast_file.item(ast_file.root_items[1]).Contract;
    try testing.expect(contract_item.is_generic);
    try testing.expectEqual(@as(usize, 1), contract_item.template_parameters.len);
    try testing.expect(contract_item.template_parameters[0].is_comptime);
}

test "compiler preserves integer comptime template metadata in AST" {
    const source_text =
        \\struct FixedPoint(comptime T: type, comptime SCALE: u256) {
        \\    raw: T,
        \\}
        \\
        \\type Scaled(comptime SCALE: u256) = FixedPoint<u256, SCALE>;
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);

    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;
    try testing.expect(struct_item.is_generic);
    try testing.expectEqual(@as(usize, 2), struct_item.template_parameters.len);
    try testing.expect(struct_item.template_parameters[0].is_comptime);
    try testing.expect(struct_item.template_parameters[1].is_comptime);

    const alias_item = ast_file.item(ast_file.root_items[1]).TypeAlias;
    try testing.expect(alias_item.is_generic);
    try testing.expectEqual(@as(usize, 1), alias_item.template_parameters.len);
    try testing.expect(alias_item.template_parameters[0].is_comptime);
}

test "compiler preserves generic bitfield and enum template metadata in AST" {
    const source_text =
        \\bitfield Flags(comptime T: type): u256 {
        \\    enabled: bool;
        \\}
        \\
        \\enum Choice(comptime T: type) {
        \\    left,
        \\    right,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);

    const bitfield_item = ast_file.item(ast_file.root_items[0]).Bitfield;
    try testing.expect(bitfield_item.is_generic);
    try testing.expectEqual(@as(usize, 1), bitfield_item.template_parameters.len);
    try testing.expect(bitfield_item.template_parameters[0].is_comptime);

    const enum_item = ast_file.item(ast_file.root_items[1]).Enum;
    try testing.expect(enum_item.is_generic);
    try testing.expectEqual(@as(usize, 1), enum_item.template_parameters.len);
    try testing.expect(enum_item.template_parameters[0].is_comptime);
}

test "compiler skips generic function templates in HIR" {
    const source_text =
        \\contract Math {
        \\    fn max(comptime T: type, a: T, b: T) -> T {
        \\        return a;
        \\    }
        \\
        \\    pub fn concrete(a: u256) -> u256 {
        \\        return a;
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @concrete"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "func.func @max"));
}

test "compiler skips generic bitfield and enum templates in HIR" {
    const source_text =
        \\bitfield Flags(comptime T: type): u256 {
        \\    enabled: bool;
        \\}
        \\
        \\enum Choice(comptime T: type) {
        \\    left,
        \\    right,
        \\}
        \\
        \\pub fn concrete() -> u256 {
        \\    return 1;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @concrete"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.bitfield_decl"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum.decl"));
}

test "compiler monomorphizes generic contract function calls in HIR" {
    const source_text =
        \\contract Math {
        \\    fn first(comptime T: type, a: T, b: T) -> T {
        \\        return a;
        \\    }
        \\
        \\    pub fn choose(a: u256, b: u256) -> u256 {
        \\        return first(u256, a, b);
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @choose"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @first__u256"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @first__u256"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "func.func @first("));
}

test "compiler monomorphizes generic contract calls with generic struct type arguments" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\contract Math {
        \\    fn first(comptime T: type, a: T, b: T) -> T {
        \\        return a;
        \\    }
        \\
        \\    pub fn choose(a: Pair<u256>, b: Pair<u256>) -> Pair<u256> {
        \\        return first(Pair<u256>, a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract_item_id = item_index.lookup("Math").?;
    const contract_item = ast_file.item(contract_item_id).Contract;
    var choose_id: ?compiler.ast.ItemId = null;
    for (contract_item.members) |member_id| {
        const member = ast_file.item(member_id).*;
        if (member != .Function) continue;
        if (std.mem.eql(u8, member.Function.name, "choose")) {
            choose_id = member_id;
            break;
        }
    }
    try testing.expect(choose_id != null);
    const choose_item = ast_file.item(choose_id.?).Function;
    const function_type = typecheck.itemLocatedType(choose_id.?).type;
    try testing.expect(function_type == .function);
    try testing.expectEqualStrings("Pair__u256", function_type.function.return_types[0].name().?);
    try testing.expectEqualStrings("Pair__u256", typecheck.body_types[choose_item.body.index()].name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @first__Pair__u256"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @first__Pair__u256"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"Pair__u256\">"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "func.func @first("));
}

test "compiler does not type-check generic call type args as runtime arguments" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\contract Math {
        \\    fn first(comptime T: type, a: T, b: T) -> T {
        \\        return a;
        \\    }
        \\
        \\    pub fn choose(a: Pair<u256>, b: Pair<u256>) -> Pair<u256> {
        \\        return first(Pair<u256>, a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const type_diags = &module_typecheck.diagnostics;
    try testing.expect(type_diags.isEmpty());
}

test "compiler type-checks runtime arguments of generic calls" {
    const source_text =
        \\contract Math {
        \\    fn first(comptime T: type, a: T, b: T) -> T {
        \\        return a;
        \\    }
        \\
        \\    pub fn choose(b: u256) -> u256 {
        \\        return first(u256, true, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(!diags.isEmpty());
    try testing.expect(diagnosticMessagesContain(diags, "expected argument type 'u256', found 'bool'"));
}

test "compiler emits user diagnostics for generic arity mismatches" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\enum Choice(comptime T: type) {
        \\    ok,
        \\}
        \\
        \\bitfield Flags(comptime T: type): u256 {
        \\    raw: T;
        \\}
        \\
        \\type Wrapper(comptime T: type) = Pair<T>;
        \\
        \\pub fn broken(
        \\    a: Pair<u256, u8>,
        \\    b: Choice<u256, u8>,
        \\    c: Flags<u256, u8>,
        \\    d: Wrapper<u256, u8>,
        \\) -> void {}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "generic struct 'Pair' expects 1 arguments, found 2"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "generic enum 'Choice' expects 1 arguments, found 2"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "generic bitfield 'Flags' expects 1 arguments, found 2"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "generic type alias 'Wrapper' expects 1 arguments, found 2"));
}

test "compiler monomorphizes integer generic contract function calls in HIR" {
    const source_text =
        \\contract Math {
        \\    fn shl_by(comptime N: u256, value: u256) -> u256 {
        \\        return value << N;
        \\    }
        \\
        \\    pub fn apply(value: u256) -> u256 {
        \\        return shl_by(8, value);
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @apply"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @shl_by__8"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @shl_by__8"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shli"));
}

test "compiler monomorphizes generic struct types on type use" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn identity(value: Pair<u256>) -> Pair<u256> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.struct_, param_type.kind());
    try testing.expectEqualStrings("Pair__u256", param_type.name().?);
    try testing.expectEqualStrings("Pair__u256", typecheck.body_types[function.body.index()].name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);
}

test "compiler monomorphizes value-parameter generic struct types on type use" {
    const source_text =
        \\struct FixedPoint(comptime T: type, comptime SCALE: u256) {
        \\    raw: T,
        \\}
        \\
        \\pub fn identity(value: FixedPoint<u256, 18>) -> FixedPoint<u256, 18> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.struct_, param_type.kind());
    try testing.expectEqualStrings("FixedPoint__u256__18", param_type.name().?);
    try testing.expectEqualStrings("FixedPoint__u256__18", typecheck.body_types[function.body.index()].name().?);
    try testing.expect(typecheck.instantiatedStructByName("FixedPoint__u256__18") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"FixedPoint__u256__18\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"FixedPoint__u256__18\">"));
}

test "compiler monomorphizes generic struct return types in non-generic functions" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn make() -> Pair<u256> {
        \\    return Pair { left: 1, right: 2 };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.body_types[function.body.index()].kind());
    try testing.expectEqualStrings("Pair__u256", typecheck.body_types[function.body.index()].name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @make"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"Pair__u256\">"));
}

test "compiler monomorphizes generic struct types nested in storage field declarations" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\contract Vault {
        \\    storage balances: map<address, Pair<u256>>;
        \\
        \\    pub fn read(user: address) -> Pair<u256> {
        \\        return balances[user];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const contract = ast_file.item(ast_file.root_items[1]).Contract;
    const field_id = contract.members[0];
    const function_id = contract.members[1];

    const field_typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = field_id });
    const field_type = field_typecheck.item_types[field_id.index()];
    try testing.expectEqual(compiler.sema.TypeKind.map, field_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.struct_, field_type.valueType().?.kind());
    try testing.expectEqualStrings("Pair__u256", field_type.valueType().?.name().?);
    try testing.expect(field_typecheck.instantiatedStructByName("Pair__u256") != null);

    const function_typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = function_id });
    const function = ast_file.item(function_id).Function;
    try testing.expectEqual(compiler.sema.TypeKind.struct_, function_typecheck.body_types[function.body.index()].kind());
    try testing.expectEqualStrings("Pair__u256", function_typecheck.body_types[function.body.index()].name().?);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.global"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_get"));
}

test "compiler monomorphizes generic struct payloads in error unions" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\error NotFound;
        \\
        \\pub fn get() -> !Pair<u256> | NotFound {
        \\    return ok(Pair { left: 1, right: 2 });
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    const function_type = typecheck.item_types[ast_file.root_items[2].index()];
    try testing.expectEqual(compiler.sema.TypeKind.function, function_type.kind());
    try testing.expectEqual(@as(usize, 1), function_type.returnTypes().len);
    const return_type = function_type.returnTypes()[0];
    try testing.expectEqual(compiler.sema.TypeKind.error_union, return_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.struct_, return_type.payloadType().?.kind());
    try testing.expectEqualStrings("Pair__u256", return_type.payloadType().?.name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.ok"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.error_union<!ora.struct<\"Pair__u256\">"));
}

test "compiler preserves generic type alias metadata in AST" {
    const source_text =
        \\type Balances(comptime K: type) = map<K, u256>;
        \\type U256 = u256;
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);

    const generic_alias = ast_file.item(ast_file.root_items[0]).TypeAlias;
    try testing.expectEqualStrings("Balances", generic_alias.name);
    try testing.expect(generic_alias.is_generic);
    try testing.expectEqual(@as(usize, 1), generic_alias.template_parameters.len);

    const plain_alias = ast_file.item(ast_file.root_items[1]).TypeAlias;
    try testing.expectEqualStrings("U256", plain_alias.name);
    try testing.expect(!plain_alias.is_generic);
}

test "compiler resolves generic type aliases through substitution" {
    const source_text =
        \\type Balances(comptime K: type) = map<K, u256>;
        \\
        \\pub fn lookup(values: Balances<address>) -> Balances<address> {
        \\    return values;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.map, param_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.address, param_type.keyType().?.kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, param_type.valueType().?.kind());
    try testing.expectEqual(compiler.sema.TypeKind.map, typecheck.body_types[function.body.index()].kind());
}

test "compiler forwards type aliases into generic struct instantiation" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\type U256Pair = Pair<u256>;
        \\
        \\pub fn identity(value: U256Pair) -> U256Pair {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    const function = ast_file.item(ast_file.root_items[2]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.struct_, param_type.kind());
    try testing.expectEqualStrings("Pair__u256", param_type.name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"Pair__u256\">"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "type_alias"));
}

test "compiler forwards integer generic aliases into value-parameter instantiation" {
    const source_text =
        \\struct FixedPoint(comptime T: type, comptime SCALE: u256) {
        \\    raw: T,
        \\}
        \\
        \\type Scaled(comptime SCALE: u256) = FixedPoint<u256, SCALE>;
        \\
        \\pub fn identity(value: Scaled<18>) -> Scaled<18> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    const function = ast_file.item(ast_file.root_items[2]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.struct_, param_type.kind());
    try testing.expectEqualStrings("FixedPoint__u256__18", param_type.name().?);
    try testing.expect(typecheck.instantiatedStructByName("FixedPoint__u256__18") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"FixedPoint__u256__18\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"FixedPoint__u256__18\">"));
}

test "compiler resolves value-generic refinement aliases through substitution" {
    const source_text =
        \\type Bounded(comptime MIN: u256, comptime MAX: u256) = InRange<u256, MIN, MAX>;
        \\
        \\pub fn clamp(value: Bounded<0, 100>) -> Bounded<0, 100> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.refinement, param_type.kind());
    try testing.expectEqualStrings("InRange", param_type.name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, param_type.refinementBaseType().?.kind());
    try testing.expectEqual(@as(usize, 3), param_type.refinement.args.len);
    try testing.expect(param_type.refinement.args[1] == .Integer);
    try testing.expectEqualStrings("0", param_type.refinement.args[1].Integer.text);
    try testing.expect(param_type.refinement.args[2] == .Integer);
    try testing.expectEqualStrings("100", param_type.refinement.args[2].Integer.text);
}

test "compiler lowers value-generic refinement aliases in HIR" {
    const source_text =
        \\type Bounded(comptime MIN: u256, comptime MAX: u256) = InRange<u256, MIN, MAX>;
        \\
        \\pub fn clamp(value: Bounded<0, 100>) -> Bounded<0, 100> {
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.in_range"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @clamp"));
}

test "compiler preserves comptime array-size template metadata in AST" {
    const source_text =
        \\type BoundedArray(comptime T: type, comptime N: u256) = [T; N];
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const alias_item = ast_file.item(ast_file.root_items[0]).TypeAlias;

    try testing.expect(alias_item.is_generic);
    try testing.expectEqual(@as(usize, 2), alias_item.template_parameters.len);

    const target = ast_file.typeExpr(alias_item.target_type);
    try testing.expect(target.* == .Array);
    try testing.expect(target.Array.size == .Name);
    try testing.expectEqualStrings("N", target.Array.size.Name.name);
}

test "compiler resolves value-generic array aliases through substitution" {
    const source_text =
        \\type BoundedArray(comptime T: type, comptime N: u256) = [T; N];
        \\
        \\pub fn identity(value: BoundedArray<u256, 10>) -> BoundedArray<u256, 10> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.array, param_type.kind());
    try testing.expectEqual(@as(?u32, 10), param_type.arrayLen());
    try testing.expectEqual(compiler.sema.TypeKind.integer, param_type.elementType().?.kind());
    try testing.expectEqual(compiler.sema.TypeKind.array, typecheck.body_types[function.body.index()].kind());
    try testing.expectEqual(@as(?u32, 10), typecheck.body_types[function.body.index()].arrayLen());
}

test "compiler lowers value-generic array aliases in HIR" {
    const source_text =
        \\type BoundedArray(comptime T: type, comptime N: u256) = [T; N];
        \\
        \\pub fn identity(value: BoundedArray<u256, 10>) -> BoundedArray<u256, 10> {
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref<10xi256>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @identity"));
}

test "compiler lowers instantiated generic struct declarations in HIR" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn identity(value: Pair<u256>) -> Pair<u256> {
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct.decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @identity"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"Pair__u256\">"));
}

test "compiler supports call-style generic types in return annotations" {
    const source_text =
        \\contract GenericPairTest {
        \\    struct Pair(comptime T: type) {
        \\        first: T;
        \\        second: T;
        \\    }
        \\
        \\    pub fn make_pair(a: u256, b: u256) -> Pair(u256) {
        \\        return Pair(u256) { first: a, second: b };
        \\    }
        \\
        \\    pub fn make_pair_u8(a: u8, b: u8) -> Pair(u8) {
        \\        return Pair(u8) { first: a, second: b };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u8") != null);
}

test "compiler monomorphizes nested generic struct types on type use" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn nested(value: Pair<Pair<u256>>) -> Pair<Pair<u256>> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.struct_, param_type.kind());
    try testing.expectEqualStrings("Pair__Pair__u256", param_type.name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);
    try testing.expect(typecheck.instantiatedStructByName("Pair__Pair__u256") != null);
}

test "compiler lowers nested generic struct instantiations in HIR" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn nested(value: Pair<Pair<u256>>) -> Pair<Pair<u256>> {
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"Pair__Pair__u256\">"));
}

test "compiler forwards nested aliases into generic struct instantiation" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\type Inner = Pair<u256>;
        \\type Outer = Pair<Inner>;
        \\
        \\pub fn nested(value: Outer) -> Outer {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[3] });
    const function = ast_file.item(ast_file.root_items[3]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.struct_, param_type.kind());
    try testing.expectEqualStrings("Pair__Pair__u256", param_type.name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);
    try testing.expect(typecheck.instantiatedStructByName("Pair__Pair__u256") != null);
}

test "compiler monomorphizes generic structs for top-level constants" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\const ZERO: Pair<u256> = Pair { left: 0, right: 0 };
        \\
        \\pub fn get() -> Pair<u256> {
        \\    return ZERO;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    const constant_type = typecheck.item_types[ast_file.root_items[1].index()];
    try testing.expectEqual(compiler.sema.TypeKind.struct_, constant_type.kind());
    try testing.expectEqualStrings("Pair__u256", constant_type.name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.const"));
}

test "compiler monomorphizes generic structs for top-level fields" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\memory current: Pair<u256>;
        \\
        \\pub fn read() -> Pair<u256> {
        \\    return current;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    const field_type = typecheck.item_types[ast_file.root_items[1].index()];
    try testing.expectEqual(compiler.sema.TypeKind.struct_, field_type.kind());
    try testing.expectEqualStrings("Pair__u256", field_type.name().?);
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Pair__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.memory.global"));
}

test "compiler monomorphizes generic enum types on type use" {
    const source_text =
        \\enum Choice(comptime T: type) {
        \\    left,
        \\    right,
        \\}
        \\
        \\pub fn identity(value: Choice<u256>) -> Choice<u256> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.enum_, param_type.kind());
    try testing.expectEqualStrings("Choice__u256", param_type.name().?);
    try testing.expectEqualStrings("Choice__u256", typecheck.body_types[function.body.index()].name().?);
    try testing.expect(typecheck.instantiatedEnumByName("Choice__u256") != null);
}

test "compiler monomorphizes value-parameter generic enum types on type use" {
    const source_text =
        \\enum Choice(comptime T: type, comptime TAG: u256) {
        \\    left,
        \\    right,
        \\}
        \\
        \\pub fn identity(value: Choice<u256, 7>) -> Choice<u256, 7> {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.enum_, param_type.kind());
    try testing.expectEqualStrings("Choice__u256__7", param_type.name().?);
    try testing.expectEqualStrings("Choice__u256__7", typecheck.body_types[function.body.index()].name().?);
    try testing.expect(typecheck.instantiatedEnumByName("Choice__u256__7") != null);
}

test "compiler lowers instantiated generic enum declarations in HIR" {
    const source_text =
        \\enum Choice(comptime T: type) {
        \\    left,
        \\    right,
        \\}
        \\
        \\pub fn identity(value: Choice<u256>) -> Choice<u256> {
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum.decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Choice__u256\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @identity"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @identity(%arg0: i256"));
}

test "compiler lowers value-parameter generic enum declarations in HIR" {
    const source_text =
        \\enum Choice(comptime T: type, comptime TAG: u256) {
        \\    left,
        \\    right,
        \\}
        \\
        \\pub fn identity(value: Choice<u256, 7>) -> Choice<u256, 7> {
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum.decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Choice__u256__7\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @identity(%arg0: i256"));
}

test "compiler monomorphizes generic bitfield types on type use" {
    const source_text =
        \\bitfield Flags(comptime T: type): u256 {
        \\    enabled: T;
        \\}
        \\
        \\pub fn read_flag(value: Flags<u8>) -> u8 {
        \\    return value.enabled;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.bitfield, param_type.kind());
    try testing.expectEqualStrings("Flags__u8", param_type.name().?);
    try testing.expect(typecheck.instantiatedBitfieldByName("Flags__u8") != null);
}

test "compiler monomorphizes value-parameter generic bitfield types on type use" {
    const source_text =
        \\bitfield Flags(comptime T: type, comptime WIDTH: u256): u256 {
        \\    raw: T;
        \\}
        \\
        \\pub fn read(value: Flags<u8, 8>) -> u8 {
        \\    return value.raw;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const function = ast_file.item(ast_file.root_items[1]).Function;

    const param_type = typecheck.pattern_types[function.parameters[0].pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.bitfield, param_type.kind());
    try testing.expectEqualStrings("Flags__u8__8", param_type.name().?);
    try testing.expect(typecheck.instantiatedBitfieldByName("Flags__u8__8") != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.body_types[function.body.index()].kind());
}

test "compiler lowers instantiated generic bitfield metadata in HIR" {
    const source_text =
        \\bitfield Flags(comptime T: type): u256 {
        \\    enabled: T;
        \\}
        \\
        \\pub fn read_flag(value: Flags<u8>) -> u8 {
        \\    return value.enabled;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Flags__u8\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bitfield"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bitfield_layout"));
}

test "compiler lowers value-parameter generic bitfield metadata in HIR" {
    const source_text =
        \\bitfield Flags(comptime T: type, comptime WIDTH: u256): u256 {
        \\    raw: T;
        \\}
        \\
        \\pub fn read(value: Flags<u8, 8>) -> u8 {
        \\    return value.raw;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"Flags__u8__8\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bitfield"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bitfield_layout"));
}

test "compiler lowers builtin, quantified, and verification expressions" {
    const source_text =
        \\pub fn verify(values: slice[u256], next: address) -> u256
        \\    ensures old(result) >= 0;
        \\{
        \\    assert(forall i: u256 where i < 4 => values[i] >= 0);
        \\    let casted = @cast(address, next);
        \\    let quotient = @divTrunc(10, 3);
        \\    let snapshot = old(quotient);
        \\    let addr = 0x1234567890abcdef1234567890abcdef12345678;
        \\    let data = hex"deadbeef";
        \\    return quotient;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    try testing.expectEqual(@as(usize, 1), function.clauses.len);
    const ensures_expr = function.clauses[0].expr;
    try testing.expect(ast_file.expression(ensures_expr).* == .Binary);

    const ensures_binary = ast_file.expression(ensures_expr).Binary;
    try testing.expect(ast_file.expression(ensures_binary.lhs).* == .Old);
    const old_result = ast_file.expression(ensures_binary.lhs).Old;
    try testing.expect(ast_file.expression(old_result.expr).* == .Name);
    try testing.expectEqualStrings("result", ast_file.expression(old_result.expr).Name.name);

    const body = ast_file.body(function.body);
    try testing.expectEqual(@as(usize, 7), body.statements.len);

    const assert_stmt = ast_file.statement(body.statements[0]).Assert;
    try testing.expect(ast_file.expression(assert_stmt.condition).* == .Quantified);
    const quantified = ast_file.expression(assert_stmt.condition).Quantified;
    try testing.expectEqual(compiler.ast.Quantifier.forall, quantified.quantifier);
    try testing.expect(ast_file.typeExpr(quantified.type_expr).* == .Path);
    try testing.expectEqualStrings("u256", ast_file.typeExpr(quantified.type_expr).Path.name);
    try testing.expect(quantified.condition != null);

    const casted_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const quotient_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const snapshot_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const addr_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const data_stmt = ast_file.statement(body.statements[5]).VariableDecl;
    const return_stmt = ast_file.statement(body.statements[6]).Return;

    try testing.expect(ast_file.expression(casted_stmt.value.?).* == .Builtin);
    try testing.expect(ast_file.expression(quotient_stmt.value.?).* == .Builtin);
    try testing.expect(ast_file.expression(snapshot_stmt.value.?).* == .Old);
    try testing.expect(ast_file.expression(addr_stmt.value.?).* == .AddressLiteral);
    try testing.expect(ast_file.expression(data_stmt.value.?).* == .BytesLiteral);
    try testing.expect(ast_file.expression(return_stmt.value.?).* == .Name);

    const casted_builtin = ast_file.expression(casted_stmt.value.?).Builtin;
    try testing.expectEqualStrings("cast", casted_builtin.name);
    try testing.expect(casted_builtin.type_arg != null);

    const quotient_builtin = ast_file.expression(quotient_stmt.value.?).Builtin;
    try testing.expectEqualStrings("divTrunc", quotient_builtin.name);
    try testing.expectEqual(@as(usize, 2), quotient_builtin.args.len);

    const resolution = try compilation.db.resolveNames(compilation.root_module_id);
    const quantified_condition = ast_file.expression(quantified.condition.?).Binary;
    try testing.expect(resolution.expr_bindings[quantified_condition.lhs.index()] != null);
    const quantified_body = ast_file.expression(quantified.body).Binary;
    const quantified_index = ast_file.expression(quantified_body.lhs).Index;
    try testing.expect(resolution.expr_bindings[quantified_index.index.index()] != null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(old_result.expr).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ensures_binary.lhs).kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(assert_stmt.condition).kind());
    try testing.expectEqual(compiler.sema.TypeKind.address, typecheck.exprType(casted_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(quotient_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(snapshot_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.address, typecheck.exprType(addr_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.bytes, typecheck.exprType(data_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[quotient_stmt.value.?.index()] != null);
    try testing.expectEqual(@as(i128, 3), try consteval.values[quotient_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler render ladder step 1 struct decl struct literal field extract" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\pub fn build() -> u256 {
        \\    let pair = Pair { first: 1, second: 2 };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
}

test "compiler render ladder step 2 add error decl" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let pair = Pair { first: 1, second: 2 };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct.decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.decl"));
}

test "compiler render ladder step 3 add array literal" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let pair = Pair { first: 1, second: 2 };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.store"));
}

test "compiler render ladder step 4 add array indexing into struct literal" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
}

test "compiler render ladder step 5 add tuple from indexed values" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
}

test "compiler render ladder switch step a single case no else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder switch step b single case plus else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder switch step c two cases plus else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder switch step d two cases no else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler renders minimal two-case switch expression" {
    const source_text =
        \\pub fn build() -> u256 {
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder step 6 add boolean switch expression" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (true) {
        \\        true => 1,
        \\        false => 2,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder step 6b add integer switch expression" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder step 7 add error constructor expression" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (true) {
        \\        true => 1,
        \\        false => 2,
        \\        else => 3,
        \\    };
        \\    let problem = Failure(7);
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return"));
}

test "compiler lowers tuple, array, struct, switch, and error return expressions" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (true) {
        \\        true => 1,
        \\        false => 2,
        \\        else => 3,
        \\    };
        \\    let problem = Failure(7);
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    try testing.expectEqual(@as(usize, 6), body.statements.len);

    const items_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const coords_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const pair_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const value_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const problem_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const return_stmt = ast_file.statement(body.statements[5]).Return;

    try testing.expect(ast_file.expression(items_stmt.value.?).* == .ArrayLiteral);
    try testing.expect(ast_file.expression(coords_stmt.value.?).* == .Tuple);
    try testing.expect(ast_file.expression(pair_stmt.value.?).* == .StructLiteral);
    try testing.expect(ast_file.expression(value_stmt.value.?).* == .Switch);
    try testing.expect(ast_file.expression(problem_stmt.value.?).* == .Call);

    const items_expr = ast_file.expression(items_stmt.value.?).ArrayLiteral;
    try testing.expectEqual(@as(usize, 3), items_expr.elements.len);

    const coords_expr = ast_file.expression(coords_stmt.value.?).Tuple;
    try testing.expectEqual(@as(usize, 2), coords_expr.elements.len);

    const pair_expr = ast_file.expression(pair_stmt.value.?).StructLiteral;
    try testing.expectEqualStrings("Pair", pair_expr.type_name);
    try testing.expectEqual(@as(usize, 2), pair_expr.fields.len);

    const value_expr = ast_file.expression(value_stmt.value.?).Switch;
    try testing.expectEqual(@as(usize, 2), value_expr.arms.len);
    try testing.expect(value_expr.else_expr != null);

    const problem_expr = ast_file.expression(problem_stmt.value.?).Call;
    try testing.expect(ast_file.expression(problem_expr.callee).* == .Name);
    try testing.expectEqualStrings("Failure", ast_file.expression(problem_expr.callee).Name.name);
    try testing.expectEqual(@as(usize, 1), problem_expr.args.len);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.array, typecheck.pattern_types[items_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.tuple, typecheck.pattern_types[coords_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.pattern_types[pair_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(return_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[value_stmt.value.?.index()] != null);
    try testing.expectEqual(@as(i128, 1), try consteval.values[value_stmt.value.?.index()].?.integer.toInt(i128));

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.error.return\""));
}

test "compiler lowers grouped struct literal bases" {
    const source_text =
        \\pub fn make() -> Pair {
        \\    let pair = (Pair){ x: 1, y: 2 };
        \\    return pair;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const pair_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const pair_expr = ast_file.expression(pair_stmt.value.?).StructLiteral;
    try testing.expectEqualStrings("Pair", pair_expr.type_name);
    try testing.expectEqual(@as(usize, 2), pair_expr.fields.len);
}

test "compiler emits AST validation diagnostics for duplicate same-scope names" {
    const source_text =
        \\pub fn helper() -> u256 {
        \\    return 1;
        \\}
        \\pub fn helper() -> u256 {
        \\    return 2;
        \\}
        \\struct Pair {
        \\    left: u256,
        \\    left: bool,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);

    try testing.expectEqual(@as(usize, 2), ast_diags.len());
    try testing.expectEqualStrings("duplicate item name 'helper' in root scope", ast_diags.items.items[0].message);
    try testing.expectEqualStrings("duplicate struct field name 'left'", ast_diags.items.items[1].message);
}

test "compiler lowers malformed syntax nodes into AST errors" {
    const source_text =
        \\pub fn run() {
        \\    let value: = 1;
        \\    let other = ;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);

    const bad_type_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expect(bad_type_stmt.type_expr != null);
    try testing.expect(ast_file.typeExpr(bad_type_stmt.type_expr.?).* == .Error);

    const bad_value_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    try testing.expect(bad_value_stmt.value != null);
    try testing.expect(ast_file.expression(bad_value_stmt.value.?).* == .Error);
}

test "compiler lowers malformed function items into AST errors" {
    const source_text =
        \\pub fn broken {
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const syntax_diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expect(!syntax_diags.isEmpty());
    const ast_file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 1), ast_file.root_items.len);
    try testing.expect(ast_file.item(ast_file.root_items[0]).* == .Error);
}

test "compiler lowers compound assignment syntax through syntax AST path" {
    const source_text =
        \\pub fn bump(x: u256) -> u256 {
        \\    let total = x;
        \\    total += 1;
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const assign = ast_file.statement(body.statements[1]).Assign;

    try testing.expectEqual(compiler.ast.AssignmentOp.add_assign, assign.op);
}

test "compiler lowers labeled block statements through syntax AST path" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    outer: {
        \\        let value = 1;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const labeled = ast_file.statement(body.statements[0]).LabeledBlock;

    try testing.expectEqualStrings("outer", labeled.label);
    try testing.expectEqual(@as(usize, 1), ast_file.body(labeled.body).statements.len);
}

test "compiler lowers labeled switch statements and jump values through syntax AST path" {
    const source_text =
        \\pub fn run(flag: u256) -> u256 {
        \\    again: switch (flag) {
        \\        0 => {
        \\            continue :again 1;
        \\        },
        \\        else => {
        \\            break :again;
        \\        }
        \\    }
        \\    return flag;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const switch_stmt = ast_file.statement(body.statements[0]).Switch;
    const continue_stmt = ast_file.statement(ast_file.body(switch_stmt.arms[0].body).statements[0]).Continue;
    const break_stmt = ast_file.statement(ast_file.body(switch_stmt.else_body.?).statements[0]).Break;

    try testing.expectEqualStrings("again", switch_stmt.label.?);
    try testing.expectEqualStrings("again", continue_stmt.label.?);
    try testing.expect(continue_stmt.value != null);
    try testing.expectEqualStrings("again", break_stmt.label.?);
}

test "compiler lowers labeled for statements through syntax AST path" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    outer: for (0..5) |i, _| {
        \\        if (i == 3) {
        \\            break :outer;
        \\        }
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const for_stmt = ast_file.statement(body.statements[0]).For;
    const if_stmt = ast_file.statement(ast_file.body(for_stmt.body).statements[0]).If;
    const break_stmt = ast_file.statement(ast_file.body(if_stmt.then_body).statements[0]).Break;

    try testing.expectEqualStrings("outer", for_stmt.label.?);
    try testing.expectEqualStrings("outer", break_stmt.label.?);
}

test "compiler lowers lock and unlock statements through syntax AST and HIR paths" {
    const source_text =
        \\contract Vault {
        \\    storage balances: u256;
        \\
        \\    pub fn run() {
        \\        @lock(balances);
        \\        @unlock(balances);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const function = ast_file.item(contract.members[1]).Function;
    const body = ast_file.body(function.body);
    try testing.expect(ast_file.statement(body.statements[0]).* == .Lock);
    try testing.expect(ast_file.statement(body.statements[1]).* == .Unlock);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock_placeholder"));
}

test "compiler emits tstore guard before guarded storage writes" {
    const source_text =
        \\contract GuardedWrites {
        \\    storage balances: u256;
        \\
        \\    pub fn touch() {
        \\        @lock(balances);
        \\        balances = 1;
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "guarded-write.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tstore.guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
}

test "compiler lowers grouped lock paths through real lock ops" {
    const source_text =
        \\contract Vault {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn run(user: address) {
        \\        @lock((balances[user]));
        \\        @unlock((balances[user]));
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock_placeholder"));
}

test "compiler lowers simple memref-backed for loops through scf.for" {
    const source_text =
        \\pub fn scan(values: slice[u256]) {
        \\    for (values) |value, index| {
        \\        assert(value >= index, "ordered");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.index_castui"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler threads carried locals through scf.for iter args" {
    const source_text =
        \\pub fn sum(values: slice[u256]) -> u256 {
        \\    let total = 0;
        \\    for (values) |value, index| {
        \\        total = total + value + index;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.index_castui"));
    try testing.expect(std.mem.count(u8, hir_text, "arith.addi") >= 2);
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers real HIR for loops with break and continue" {
    const source_text =
        \\pub fn count(values: slice[u256]) -> u256 {
        \\    let continued = 0;
        \\    for (values) |value| {
        \\        continued = continued + value;
        \\        continue;
        \\    }
        \\    let stopped = 0;
        \\    for (values) |value| {
        \\        stopped = stopped + value;
        \\        break;
        \\    }
        \\    return continued + stopped;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers integer range for loops with break and continue" {
    const source_text =
        \\pub fn run() {
        \\    for (0..20) |i, _| {
        \\        if (i > 10) {
        \\            break;
        \\        }
        \\    }
        \\    for (0..10) |i, _| {
        \\        if (i % 2 == 0) {
        \\            continue;
        \\        }
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers for invariants through ora.invariant" {
    const source_text =
        \\pub fn scan(values: slice[u256]) {
        \\    for (values) |value, index|
        \\        invariant value >= index;
        \\    {
        \\        assert(value >= index, "ordered");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.invariant"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers named contract invariants through ora.invariant" {
    const source_text =
        \\contract Counter {
        \\    storage var value: u256;
        \\    invariant value_nonnegative(value >= 0);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.invariant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.cmpi"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.call\""));
}

test "compiler lowers for loops with early return without placeholders" {
    const source_text =
        \\pub fn scan(values: slice[u256], stop_at: u256) -> u256 {
        \\    let total = 0;
        \\    for (values) |value| {
        \\        if (value == stop_at) {
        \\            return total;
        \\        }
        \\        total = total + value;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.conditional_return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler skips unknown carried locals in for lowering" {
    const source_text =
        \\pub fn scan(values: slice[u256]) -> u256 {
        \\    let total = 0;
        \\    let bad = total.missing;
        \\    for (values) |value| {
        \\        bad = value.missing;
        \\        total = total + value;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers for loops with typed carried locals without explicit initializer" {
    const source_text =
        \\pub fn sum(values: slice[u256]) -> u256 {
        \\    let total: u256;
        \\    for (values) |value| {
        \\        total = total + value;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers while loops with nested for carried locals" {
    const source_text =
        \\pub fn sum(limit: u256, values: slice[u256]) -> u256 {
        \\    let total = 0;
        \\    let i = 0;
        \\    while (i < limit) {
        \\        for (values) |value| {
        \\            total = total + value;
        \\        }
        \\        i = i + 1;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers for loops with nested try carried locals" {
    const source_text =
        \\pub fn sum(values: slice[u256]) -> u256 {
        \\    let total = 0;
        \\    for (values) |value| {
        \\        try {
        \\            total = total + value;
        \\        } catch (err) {
        \\            total = total + 1;
        \\        }
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
}

test "compiler lowers direct map index load and store through real map ops" {
    const source_text =
        \\contract Maps {
        \\    storage table: map<u256, u256>;
        \\
        \\    pub fn touch() -> u256 {
        \\        table[1] = 2;
        \\        return table[1];
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "maps.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_store"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_get"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.index_access"));
}

test "compiler lowers wrapping add through real wrapping op" {
    const source_text =
        \\pub fn wrap(a: u256, b: u256) -> u256 {
        \\    return a +% b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.add_wrapping"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.add_wrapping\""));
}

test "compiler lowers remaining wrapping ops through real wrapping ops" {
    const source_text =
        \\pub fn wrap_all(a: u256, b: u256, c: u256) -> u256 {
        \\    let s1 = a -% b;
        \\    let s2 = a *% b;
        \\    let s3 = a <<% c;
        \\    let s4 = a >>% c;
        \\    return s1 +% s2 +% s3 +% s4;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sub_wrapping"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.mul_wrapping"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.shl_wrapping"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.shr_wrapping"));
}

test "compiler lowers wrapping power through ora.power" {
    const source_text =
        \\pub fn wrap_pow(a: u256, b: u256) -> u256 {
        \\    return a **% b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.power"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.power\""));
}

test "compiler lowers checked power with overflow assert" {
    const source_text =
        \\pub fn checked_pow(a: u256, b: u256) -> u256 {
        \\    return a ** b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.power"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.power\""));
}

test "compiler lowers bitwise not through xor mask" {
    const source_text =
        \\pub fn invert(value: u256) -> u256 {
        \\    return ~value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.xori"));
}

test "compiler lowers bitwise and compound assignment" {
    const source_text =
        \\pub fn mask(input: u256, mask: u256) -> u256 {
        \\    var value = input;
        \\    value &= mask;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.andi"));
}

test "compiler lowers bitwise or and xor compound assignment" {
    const source_text =
        \\pub fn mix(input: u256, mask: u256, toggles: u256) -> u256 {
        \\    var value = input;
        \\    value |= mask;
        \\    value ^= toggles;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.ori"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.xori"));
}

test "compiler lowers shift compound assignment" {
    const source_text =
        \\pub fn shift_ops(input: u256, amount: u256) -> u256 {
        \\    var value = input;
        \\    value <<= amount;
        \\    value >>= amount;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shli"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shrsi"));
}

test "compiler lowers power and wrapping compound assignment" {
    const source_text =
        \\pub fn update_all(input: u256, exp: u256, delta: u256) -> u256 {
        \\    var value = input;
        \\    value **= exp;
        \\    value +%= delta;
        \\    value -%= delta;
        \\    value *%= delta;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.power"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.add_wrapping"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sub_wrapping"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.mul_wrapping"));
}

test "compiler lowers checked arithmetic compound assignment" {
    const source_text =
        \\pub fn update_checked(input: u256, delta: u256, divisor: u256) -> u256 {
        \\    var value = input;
        \\    value += delta;
        \\    value -= delta;
        \\    value *= delta;
        \\    value /= divisor;
        \\    value %= divisor;
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.addi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.subi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.muli"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.divsi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.remsi"));
}

test "compiler rethreads nested map assignment to outer map" {
    const source_text =
        \\contract Test {
        \\    storage allowances: map<address, map<address, u256>>;
        \\
        \\    pub fn setAllowance(owner: address, spender: address, amount: u256) {
        \\        allowances[owner][spender] = amount;
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "nested-map-store.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.count(u8, hir_text, "ora.map_store") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_get"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.index_access"));
}

test "compiler lowers slice index load and store through memref ops" {
    const source_text =
        \\pub fn touch(values: slice[u256]) -> u256 {
        \\    values[0] = 7;
        \\    return values[0];
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "slice-index.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.store"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.index_castui"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref<?xi256>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.index_access"));
}

test "compiler lowers array literals through memref allocation and stores" {
    const source_text =
        \\pub fn read_first() -> u256 {
        \\    let items = [1, 2, 3];
        \\    return items[0];
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "array-literal.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(std.mem.count(u8, hir_text, "memref.store") >= 3);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref<3xi256>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.array.create"));
}

test "compiler lowers destructuring bindings through struct field extracts" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn sum() -> u256 {
        \\    let pair = Pair { left: 1, right: 2 };
        \\    let .{ left: a, right: b } = pair;
        \\    return a + b;
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "destructure-bind.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.count(u8, hir_text, "ora.struct_field_extract") >= 2);
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers destructuring assignment through struct field extracts" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn sum() -> u256 {
        \\    let left = 0;
        \\    let right = 0;
        \\    .{ left, right } = Pair { left: 4, right: 5 };
        \\    return left + right;
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "destructure-assign.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.count(u8, hir_text, "ora.struct_field_extract") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.addi"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers fallback break and continue through real ops" {
    const source_text =
        \\pub fn stop() {
        \\    break;
        \\    continue;
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "fallback-jumps.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.break"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.continue"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.continue\""));
}

test "compiler lowers comptime block expressions through syntax AST path" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let value = 1;
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret = ast_file.statement(body.statements[0]).Return;
    try testing.expect(ret.value != null);
    const comptime_expr = ast_file.expression(ret.value.?).Comptime;
    try testing.expectEqual(@as(usize, 2), ast_file.body(comptime_expr.body).statements.len);
}

test "compiler keeps malformed declaration shapes on syntax lowering path" {
    const source_text =
        \\struct Pair {
        \\    left:;
        \\}
        \\
        \\pub fn broken(value) -> u256 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(!ast_diags.isEmpty());

    const ast_file = try compilation.db.astFile(module.file_id);
    const pair_item = ast_file.item(ast_file.root_items[0]).Struct;
    try testing.expect(ast_file.typeExpr(pair_item.fields[0].type_expr).* == .Error);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    try testing.expect(ast_file.typeExpr(function.parameters[0].type_expr).* == .Error);
}

test "compiler reports sema diagnostics for unresolved names and invalid operations" {
    const source_text =
        \\pub fn broken(flag: bool, value: u256) -> u256 {
        \\    let missing = nope;
        \\    let bad_not = !value;
        \\    let bad_add = flag + value;
        \\    let bad_field = value.balance;
        \\    let bad_index = flag[0];
        \\    let bad_call = value();
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;

    const resolution_diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), resolution_diags.len());
    try testing.expectEqualStrings("undefined name 'nope'", resolution_diags.items.items[0].message);

    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqualStrings("invalid unary operator '!' for type 'u256'", type_diags.items.items[0].message);
    try testing.expectEqualStrings("invalid binary operator '+' for types 'bool' and 'u256'", type_diags.items.items[1].message);
    try testing.expectEqualStrings("type 'u256' has no field 'balance'", type_diags.items.items[2].message);
    try testing.expectEqualStrings("type 'bool' is not indexable", type_diags.items.items[3].message);
    try testing.expectEqualStrings("type 'u256' is not callable", type_diags.items.items[4].message);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const body = ast_file.body(function.body);
    const missing_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const bad_not_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const bad_add_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const bad_field_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const bad_index_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const bad_call_stmt = ast_file.statement(body.statements[5]).VariableDecl;

    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[missing_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_not_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_add_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_field_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_index_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_call_stmt.pattern.index()].kind());
}

test "compiler suppresses cascading diagnostics from unknown expressions" {
    const source_text =
        \\pub fn broken() -> u256 {
        \\    let value = missing.field + 1;
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;

    const resolution_diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), resolution_diags.len());
    try testing.expectEqualStrings("undefined name 'missing'", resolution_diags.items.items[0].message);

    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 0), type_diags.len());

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[value_stmt.pattern.index()].kind());
}

test "compiler reports sema diagnostics for declaration assignment and return mismatches" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256 = true;
        \\}
        \\
        \\const LIMIT: u256 = false;
        \\
        \\pub fn broken(flag: bool) -> u256 {
        \\    let a: u256 = flag;
        \\    let b = 1;
        \\    b = flag;
        \\    return flag;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqualStrings("field 'total' expects type 'u256', found 'bool'", type_diags.items.items[0].message);
    try testing.expectEqualStrings("constant 'LIMIT' expects type 'u256', found 'bool'", type_diags.items.items[1].message);
    try testing.expectEqualStrings("declaration expects type 'u256', found 'bool'", type_diags.items.items[2].message);
    try testing.expectEqualStrings("assignment expects type 'integer', found 'bool'", type_diags.items.items[3].message);
    try testing.expectEqualStrings("return expects type 'u256', found 'bool'", type_diags.items.items[4].message);

    const function = ast_file.item(ast_file.root_items[2]).Function;
    const full_typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    const body = ast_file.body(function.body);
    const a_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[3]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.integer, full_typecheck.pattern_types[a_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, full_typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler reports sema diagnostics for control flow conditions and switch branch mismatches" {
    const source_text =
        \\pub fn broken(flag: u256) -> u256 {
        \\    if (1) {
        \\        let a = 1;
        \\    }
        \\    while (2) {
        \\        break;
        \\    }
        \\    assert(3);
        \\    assume(4);
        \\    let value = switch (flag) {
        \\        0 => 1,
        \\        1 => false,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqualStrings("if condition must be 'bool', found 'integer'", type_diags.items.items[0].message);
    try testing.expectEqualStrings("while condition must be 'bool', found 'integer'", type_diags.items.items[1].message);
    try testing.expectEqualStrings("assert condition must be 'bool', found 'integer'", type_diags.items.items[2].message);
    try testing.expectEqualStrings("assume condition must be 'bool', found 'integer'", type_diags.items.items[3].message);
    try testing.expectEqualStrings("switch expression branches have incompatible types 'integer' and 'bool'", type_diags.items.items[4].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[5]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler merges compatible integer branch types to the wider integer" {
    const source_text =
        \\pub fn widen(flag: bool, small: u8, big: u256) -> u256 {
        \\    let value = switch (flag) {
        \\        true => small,
        \\        else => big,
        \\    };
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(type_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[1]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqualStrings("u256", typecheck.pattern_types[value_stmt.pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqualStrings("u256", typecheck.exprType(ret_stmt.value.?).name().?);
}

test "compiler types and lowers all-constant switch expressions without else" {
    const source_text =
        \\pub fn choose(tag: u256) -> u256 {
        \\    let value = switch (tag) {
        \\        0 => 1,
        \\        1 => 2,
        \\    };
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 0), type_diags.len());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[1]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.switch_expr\""));
}

test "compiler reports heterogeneous array literals and keeps tuple element types" {
    const source_text =
        \\pub fn build(flag: bool, small: u8, big: u256) -> bool {
        \\    let ok = [1, 2, 3];
        \\    let widened = [small, big, 3];
        \\    let bad = [1, false];
        \\    let pair = (flag, 7);
        \\    return pair[0];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 1), type_diags.len());
    try testing.expectEqualStrings("array literal elements have incompatible types 'integer' and 'bool'", type_diags.items.items[0].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    const ok_pattern = findVariablePatternByName(ast_file, body.statements, "ok").?;
    const widened_pattern = findVariablePatternByName(ast_file, body.statements, "widened").?;

    const ok_type = typecheck.pattern_types[ok_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.array, ok_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, ok_type.elementType().?.kind());

    const widened_type = typecheck.pattern_types[widened_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.array, widened_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, widened_type.elementType().?.kind());
    try testing.expectEqualStrings("u256", widened_type.elementType().?.name().?);

    const pair_pattern = findVariablePatternByName(ast_file, body.statements, "pair").?;
    const pair_type = typecheck.pattern_types[pair_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.tuple, pair_type.kind());
    try testing.expectEqual(@as(usize, 2), pair_type.tupleTypes().len);
    try testing.expectEqual(compiler.sema.TypeKind.bool, pair_type.tupleTypes()[0].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, pair_type.tupleTypes()[1].kind());
    const ret_stmt = ast_file.statement(body.statements[body.statements.len - 1]).Return;
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler assigns integer array literals to concrete integer array types" {
    const source_text =
        \\pub fn build() -> [u256; 4] {
        \\    let dest: [u256; 4] = [0, 0, 0, 0];
        \\    return dest;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler rejects integer array literals assigned to bool arrays" {
    const source_text =
        \\pub fn build() -> [bool; 2] {
        \\    let dest: [bool; 2] = [0, 0];
        \\    return dest;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "declaration expects type 'array', found 'array'"));
}

test "compiler rejects array length mismatches in assignments" {
    const source_text =
        \\pub fn build() -> [u256; 3] {
        \\    let dest: [u256; 3] = [0, 0, 0, 0];
        \\    return dest;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "declaration expects type 'array', found 'array'"));
}

test "compiler rejects log statements with wrong arity" {
    const source_text =
        \\contract C {
        \\    log Transfer(from: address, amount: u256);
        \\
        \\    pub fn run(addr: address) {
        \\        log Transfer(addr);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "log 'Transfer' expects 2 arguments, found 1"));
}

test "compiler rejects log declarations with more than three indexed fields" {
    const source_text =
        \\contract C {
        \\    log TooMany(indexed a: address, indexed b: address, indexed c: address, indexed d: address);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "log declarations support at most 3 indexed fields"));
}

test "compiler rejects struct-typed indexed log fields" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\contract C {
        \\    log Bad(indexed p: Pair);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "indexed log field 'p' has unsupported type 'Pair'"));
}

test "compiler rejects lock and unlock builtins in expression position" {
    const source_text =
        \\contract Locked {
        \\    storage var balances: map<address, u256>;
        \\
        \\    pub fn bad_lock(user: address) -> u256 {
        \\        let tmp = @lock(balances[user]);
        \\        return tmp;
        \\    }
        \\
        \\    pub fn bad_unlock(user: address) -> bool {
        \\        @lock(balances[user]);
        \\        return @unlock(balances[user]);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!typecheck.diagnostics.isEmpty());
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "@lock is statement-only and cannot be used in expression position"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "@unlock is statement-only and cannot be used in expression position"));
}

test "compiler contextualizes nested array literals to their declared element types" {
    const source_text =
        \\pub fn nested() -> [[u256; 2]; 2] {
        \\    let inner1: [u256; 2] = [1, 2];
        \\    let inner2: [u256; 2] = [3, 4];
        \\    return [inner1, inner2];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @nested() -> memref<2xmemref<2xi256>>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca() : memref<2xmemref<2xi256>>"));
}

test "compiler lowers enum fields inside struct declarations to integer wire types" {
    const source_text =
        \\enum Status { A, B }
        \\
        \\struct Entry {
        \\    status: Status,
        \\    amount: u256,
        \\}
        \\
        \\contract C {
        \\    storage var entry: Entry;
        \\
        \\    pub fn update(status: Status) {
        \\        let next: Entry = .{ .status = status, .amount = 1 };
        \\        entry = next;
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.field_types = [i256, i256]"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct<\"Status\">"));
}

test "compiler lowers std coinbase and msg value builtins" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Builtins {
        \\    pub fn coinbase() -> address {
        \\        return std.block.coinbase();
        \\    }
        \\
        \\    pub fn value() -> u256 {
        \\        return std.msg.value();
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.evm.coinbase : !ora.address"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.evm.callvalue : i256"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.return %c0_i256 : i256"));
}

test "compiler uses const-evaluated tuple indices during type checking" {
    const source_text =
        \\pub fn pick(flag: bool) -> bool {
        \\    let pair = (flag, 7);
        \\    return pair[1 - 1];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(type_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[1]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler reports invalid constant shift amounts" {
    const source_text =
        \\pub fn shift(v: u8) -> u8 {
        \\    let ok = v << 7;
        \\    let bad = v << 8;
        \\    return ok;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 1), type_diags.len());
    try testing.expectEqualStrings("shift amount 8 out of range for type 'u8'", type_diags.items.items[0].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const ok_pattern = findVariablePatternByName(ast_file, body.statements, "ok").?;
    const bad_pattern = findVariablePatternByName(ast_file, body.statements, "bad").?;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[ok_pattern.index()].kind());
    try testing.expectEqualStrings("u8", typecheck.pattern_types[ok_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_pattern.index()].kind());
}

test "compiler reports integer constant overflow against declared widths" {
    const source_text =
        \\storage var total: u8 = 256;
        \\const LIMIT: u8 = 256;
        \\pub fn narrow() -> u8 {
        \\    let a: u8 = 256;
        \\    let b: u8 = 1;
        \\    b = 256;
        \\    return 256;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqual(@as(usize, 5), countDiagnosticMessages(type_diags, "constant value 256 does not fit in type 'u8'"));
}

test "compiler reports integer constant overflow at call sites" {
    const source_text =
        \\fn take(value: u8) -> u8 {
        \\    return value;
        \\}
        \\
        \\pub fn narrow() -> u8 {
        \\    return take(256);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expect(countDiagnosticMessages(type_diags, "constant value 256 does not fit in type 'u8'") >= 1);
}

test "compiler reports constant cast overflow against target integer widths" {
    const source_text =
        \\pub fn casted() -> u8 {
        \\    let ok = @cast(u8, 255);
        \\    let bad = @cast(u8, 256);
        \\    return ok;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 1), type_diags.len());
    try testing.expectEqualStrings("constant value 256 does not fit in cast target type 'u8'", type_diags.items.items[0].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const ok_pattern = findVariablePatternByName(ast_file, body.statements, "ok").?;
    const bad_pattern = findVariablePatternByName(ast_file, body.statements, "bad").?;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[ok_pattern.index()].kind());
    try testing.expectEqualStrings("u8", typecheck.pattern_types[ok_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_pattern.index()].kind());
}

test "compiler lowers builtin cast through real conversion ops" {
    const source_text =
        \\pub fn casts(value: u256, raw: u160) -> address {
        \\    let narrowed = @cast(u8, value);
        \\    let widened = @cast(u256, narrowed);
        \\    let addr = @cast(address, raw);
        \\    return addr;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.trunci"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.extui"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.i160.to.addr"));
}

test "compiler lowers builtin bitCast through real bitcast op" {
    const source_text =
        \\pub fn recast(value: u160) -> address {
        \\    let same = @bitCast(address, value);
        \\    return same;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;
    const builtin = ast_file.expression(decl.value.?).Builtin;
    try testing.expectEqualStrings("bitCast", builtin.name);
    try testing.expect(builtin.type_arg != null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(compiler.sema.TypeKind.address, typecheck.exprType(decl.value.?).kind());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.bitcast"));
}

test "compiler lowers builtin truncate through unchecked truncation" {
    const source_text =
        \\pub fn shrink(value: u256) -> u8 {
        \\    let narrowed = @truncate(u8, value);
        \\    return narrowed;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;
    const builtin = ast_file.expression(decl.value.?).Builtin;
    try testing.expectEqualStrings("truncate", builtin.name);
    try testing.expect(builtin.type_arg != null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(decl.value.?).kind());
    try testing.expectEqualStrings("u8", typecheck.exprType(decl.value.?).name().?);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.trunci"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "safe cast narrowing overflow"));
}

test "compiler lowers overflow builtins through real tuple results" {
    const source_text =
        \\pub fn overflow_ops(a: u8, b: u8) -> bool {
        \\    let added = @addWithOverflow(a, b);
        \\    let negated = @negWithOverflow(a);
        \\    let divided = @divWithOverflow(a, b);
        \\    let powered = @powerWithOverflow(a, b);
        \\    return added[1] || negated[1] || divided[1] || powered[1];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    const added_pattern = findVariablePatternByName(ast_file, body.statements, "added").?;
    const added_type = typecheck.pattern_types[added_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.tuple, added_type.kind());
    try testing.expectEqual(@as(usize, 2), added_type.tupleTypes().len);
    try testing.expectEqual(compiler.sema.TypeKind.integer, added_type.tupleTypes()[0].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, added_type.tupleTypes()[1].kind());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 4, "ora.tuple_create"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.addi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.subi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.divui"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.power"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.tuple.create\""));
}

test "compiler supports named field access on overflow builtin results" {
    const source_text =
        \\pub fn overflow_fields(a: u8, b: u8) -> bool {
        \\    let added = @addWithOverflow(a, b);
        \\    let subbed = @subWithOverflow(a, b);
        \\    let mulled = @mulWithOverflow(a, b);
        \\    return added.value == a && !added.overflow && subbed.value == a && mulled.overflow == mulled[1];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    const added_pattern = findVariablePatternByName(ast_file, body.statements, "added").?;
    const added_type = typecheck.pattern_types[added_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.tuple, added_type.kind());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 4, "ora.tuple_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler assigns overflow builtin results to overflow record types" {
    const source_text =
        \\pub fn run(rate_per_second: u256, duration: u256) -> u256 {
        \\    let expected_total: struct { value: u256, overflow: bool } = @mulWithOverflow(rate_per_second, duration);
        \\    if (expected_total.overflow) {
        \\        return 0;
        \\    }
        \\    return expected_total.value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_extract"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.muli"));
}

test "compiler supports general anonymous struct types" {
    const source_text =
        \\pub fn run(amount: u256) -> u256 {
        \\    let payload: struct { amount: u256, ok: bool } = .{ .amount = amount, .ok = true };
        \\    if (payload.ok) {
        \\        return payload.amount;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(typecheck.diagnostics.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const payload_pattern = findVariablePatternByName(ast_file, body.statements, "payload").?;
    const payload_type = typecheck.pattern_types[payload_pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.anonymous_struct, payload_type.kind());
    try testing.expectEqual(@as(usize, 2), payload_type.anonymous_struct.fields.len);
    try testing.expectEqualStrings("amount", payload_type.anonymous_struct.fields[0].name);
    try testing.expectEqualStrings("ok", payload_type.anonymous_struct.fields[1].name);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.tuple_extract"));
}

test "compiler const eval preserves integers wider than i128" {
    const source_text =
        \\pub fn huge() -> u256 {
        \\    return 340282366920938463463374607431768211456;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    const value = consteval.values[ret_stmt.value.?.index()].?.integer;
    const text = try value.toString(testing.allocator, 10, .lower);
    defer testing.allocator.free(text);

    try testing.expectEqualStrings("340282366920938463463374607431768211456", text);

    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 0), type_diags.len());
}

test "compiler const eval resolves local name bindings in sequence" {
    const source_text =
        \\pub fn fold() -> u256 {
        \\    let base = 2;
        \\    let sum = base + 3;
        \\    return sum;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const base_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const sum_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[2]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 2), try consteval.values[base_stmt.value.?.index()].?.integer.toInt(i128));
    try testing.expectEqual(@as(i128, 5), try consteval.values[sum_stmt.value.?.index()].?.integer.toInt(i128));
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval respects exclusive switch range patterns" {
    const source_text =
        \\pub fn choose() -> u256 {
        \\    let value = switch (2) {
        \\        1..2 => 7,
        \\        else => 9,
        \\    };
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[0]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[value_stmt.value.?.index()] != null);
    try testing.expectEqual(@as(i128, 9), try consteval.values[value_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval sequences comptime block locals" {
    const source_text =
        \\pub fn choose() -> u256 {
        \\    return comptime {
        \\        let value = 1;
        \\        let next = value + 4;
        \\        next;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes multi statement comptime scalar calls" {
    const source_text =
        \\comptime fn helper() -> u256 {
        \\    let base = 2;
        \\    let sum = base + 3;
        \\    return sum;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        helper();
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes multi statement comptime aggregate calls" {
    const source_text =
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\comptime fn make() -> Box {
        \\    let value = 42;
        \\    return Box { value: value };
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let built = make();
        \\        built.value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 42), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval wraps typed integer declarations for wrapping ops" {
    const source_text =
        \\pub fn run() -> u8 {
        \\    const x: u8 = 255 +% 1;
        \\    return x;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[1]).Return;

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 0), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval preserves signed negative literals without wrapping" {
    const source_text =
        \\pub fn run() -> i8 {
        \\    const x: i8 = -1;
        \\    return x;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const decl_stmt = ast_file.statement(body.statements[0]).VariableDecl;

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, -1), try consteval.values[decl_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler allows local name after early-return if" {
    const source_text =
        \\contract EarlyReturn {
        \\    pub fn process(x: u256) -> u256 {
        \\        if (x > 1000) {
        \\            return 1000;
        \\        }
        \\
        \\        var output: u256 = x * 2;
        \\        if (output > 500) {
        \\            return 500;
        \\        }
        \\
        \\        output += 10;
        \\        return output;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());

    _ = try compilation.db.lowerToHir(compilation.root_module_id);
}

test "compiler const eval bridges comptime string values into sema results" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let name = "ERC20";
        \\        name.len;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const name_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualStrings("ERC20", consteval.values[name_decl.value.?.index()].?.string);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval can populate callee type checks on demand" {
    const source_text =
        \\comptime fn helper() -> u256 {
        \\    return 1;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        helper();
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const helper_id = ast_file.root_items[0];
    const cache = &compilation.db.typecheck_slots.items[compilation.root_module_id.index()];

    try testing.expectEqual(@as(usize, 0), cache.entries.count());

    _ = try compilation.db.constEval(compilation.root_module_id);
    const populated_count = cache.entries.count();
    try testing.expect(populated_count > 0);

    _ = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = helper_id });
    try testing.expectEqual(populated_count, cache.entries.count());
}

test "compiler const eval instantiates generic type values in comptime blocks" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        Pair<u256>;
        \\        7;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const type_stmt = ast_file.statement(comptime_body.statements[0]).Expr;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[type_stmt.expr.index()] == null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const instantiated = typecheck.instantiatedStructByName("Pair__u256");
    try testing.expect(instantiated != null);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.type_value"));
}

test "compiler const eval resolves generic type aliases in comptime blocks" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\type U256Pair = Pair<u256>;
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        U256Pair;
        \\        9;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const type_stmt = ast_file.statement(comptime_body.statements[0]).Expr;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[type_stmt.expr.index()] == null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);
}

test "compiler const eval instantiates generic struct literals in comptime blocks" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let pair = Pair<u256> { left: 1, right: 2 };
        \\        pair.left;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const pair_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const field_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqualStrings("Pair__u256", typecheck.exprType(pair_decl.value.?).name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(field_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 1), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
    try testing.expect(typecheck.instantiatedStructByName("Pair__u256") != null);
}

test "compiler const eval generic calls propagate comptime type bindings" {
    const source_text =
        \\comptime fn id(comptime T: type, value: T) -> T {
        \\    return value;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        id(u256, 2);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 2), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls accept generic instantiated arguments" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\comptime fn project_right(comptime T: type, pair: Pair<T>) -> T {
        \\    return pair.right;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        project_right(u256, Pair<u256> { left: 1, right: 2 });
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 2), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls can return generic struct values" {
    const source_text =
        \\struct Box(comptime T: type) {
        \\    value: T,
        \\}
        \\
        \\comptime fn make_box(comptime T: type, value: T) -> Box<T> {
        \\    return Box<T> { value: value };
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let box = make_box(u256, 42);
        \\        box.value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const box_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const field_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.exprType(box_decl.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(field_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 42), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls propagate comptime integer bindings" {
    const source_text =
        \\comptime fn shl_by(comptime N: u256, value: u256) -> u256 {
        \\    return value << N;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        shl_by(8, 1);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 256), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls can return value-generic struct values" {
    const source_text =
        \\struct FixedPoint(comptime T: type, comptime SCALE: u256) {
        \\    value: T,
        \\}
        \\
        \\comptime fn make_fixed(comptime T: type, comptime SCALE: u256, value: T) -> FixedPoint<T, SCALE> {
        \\    return FixedPoint<T, SCALE> { value: value };
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let fp = make_fixed(u256, 18, 42);
        \\        fp.value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const fp_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const field_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.exprType(fp_decl.value.?).kind());
    try testing.expectEqualStrings("FixedPoint__u256__18", typecheck.exprType(fp_decl.value.?).name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(field_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 42), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval resolves value-generic aliases in comptime blocks" {
    const source_text =
        \\struct FixedPoint(comptime T: type, comptime SCALE: u256) {
        \\    value: T,
        \\}
        \\
        \\type Scaled(comptime SCALE: u256) = FixedPoint<u256, SCALE>;
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        Scaled<18>;
        \\        1;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expect(typecheck.instantiatedStructByName("FixedPoint__u256__18") != null);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 1), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls can return value-generic alias values" {
    const source_text =
        \\struct FixedPoint(comptime T: type, comptime SCALE: u256) {
        \\    value: T,
        \\}
        \\
        \\type Scaled(comptime SCALE: u256) = FixedPoint<u256, SCALE>;
        \\
        \\comptime fn make_scaled(comptime SCALE: u256, value: u256) -> Scaled<SCALE> {
        \\    return Scaled<SCALE> { value: value };
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let fp = make_scaled(18, 42);
        \\        fp.value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[3]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const fp_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const field_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[3] });
    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.exprType(fp_decl.value.?).kind());
    try testing.expectEqualStrings("FixedPoint__u256__18", typecheck.exprType(fp_decl.value.?).name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(field_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 42), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval resolves array-size generic aliases in comptime blocks" {
    const source_text =
        \\type BoundedArray(comptime T: type, comptime N: u256) = [T; N];
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let values: BoundedArray<u256, 3> = [1, 2, 3];
        \\        values[1];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const values_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const index_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqual(compiler.sema.TypeKind.array, typecheck.exprType(values_decl.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(index_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 2), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls can return array-size generic alias values" {
    const source_text =
        \\type BoundedArray(comptime T: type, comptime N: u256) = [T; N];
        \\
        \\comptime fn make_bounded(comptime T: type, comptime N: u256, a: T, b: T, c: T) -> BoundedArray<T, N> {
        \\    return [a, b, c];
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let values = make_bounded(u256, 3, 4, 5, 6);
        \\        values[2];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const values_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const index_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.array, typecheck.exprType(values_decl.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(index_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 6), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval resolves value-generic refinement aliases in comptime blocks" {
    const source_text =
        \\type Bounded(comptime MIN: u256, comptime MAX: u256) = InRange<u256, MIN, MAX>;
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        Bounded<0, 100>;
        \\        1;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 1), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls can return value-generic refinement alias values" {
    const source_text =
        \\type Bounded(comptime MIN: u256, comptime MAX: u256) = InRange<u256, MIN, MAX>;
        \\
        \\comptime fn make_bounded(comptime MIN: u256, comptime MAX: u256, value: u256) -> Bounded<MIN, MAX> {
        \\    return value;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let value = make_bounded(0, 100, 42);
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const value_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const value_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.refinement, typecheck.exprType(value_decl.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.refinement, typecheck.exprType(value_stmt.expr).kind());
    try testing.expectEqualStrings("InRange", typecheck.exprType(value_decl.value.?).name().?);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 42), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls can return nested generic struct values" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\comptime fn make_nested(comptime T: type, left: T, right: T) -> Pair<Pair<T>> {
        \\    return Pair<Pair<T>> {
        \\        left: Pair<T> { left: left, right: right },
        \\        right: Pair<T> { left: right, right: left },
        \\    };
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let nested = make_nested(u256, 1, 2);
        \\        nested.right.left;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const nested_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const field_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.exprType(nested_decl.value.?).kind());
    try testing.expectEqualStrings("Pair__Pair__u256", typecheck.exprType(nested_decl.value.?).name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(field_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 2), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval generic calls can return nested value-generic alias values" {
    const source_text =
        \\struct FixedPoint(comptime T: type, comptime SCALE: u256) {
        \\    value: T,
        \\}
        \\
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\type Scaled(comptime SCALE: u256) = FixedPoint<u256, SCALE>;
        \\
        \\comptime fn make_scaled_pair(comptime SCALE: u256, left: u256, right: u256) -> Pair<Scaled<SCALE>> {
        \\    return Pair<Scaled<SCALE>> {
        \\        left: Scaled<SCALE> { value: left },
        \\        right: Scaled<SCALE> { value: right },
        \\    };
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let pair = make_scaled_pair(18, 4, 5);
        \\        pair.right.value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[4]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_expr = ast_file.expression(ret_stmt.value.?).Comptime;
    const comptime_body = ast_file.body(comptime_expr.body);
    const pair_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;
    const field_stmt = ast_file.statement(comptime_body.statements[1]).Expr;

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[4] });
    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.exprType(pair_decl.value.?).kind());
    try testing.expectEqualStrings("Pair__FixedPoint__u256__18", typecheck.exprType(pair_decl.value.?).name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(field_stmt.expr).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler db breaks same-key const eval recursion with unknown sentinel" {
    const source_text =
        \\comptime fn loop() -> u256 {
        \\    return loop();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);

    _ = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    const consteval_diags = try compilation.db.constEvalDiagnostics(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(consteval_diags, "comptime recursion depth exceeded"));
}

test "compiler const eval executes comptime if and assignment statements" {
    const source_text =
        \\pub fn choose() -> u256 {
        \\    return comptime {
        \\        let value = 1;
        \\        if (true) {
        \\            value += 4;
        \\        } else {
        \\            value = 99;
        \\        }
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes comptime switch statements" {
    const source_text =
        \\pub fn choose() -> u256 {
        \\    return comptime {
        \\        let value = 0;
        \\        switch (2) {
        \\            0 => {
        \\                value = 10;
        \\            }
        \\            1..2 => {
        \\                value = 20;
        \\            }
        \\            else => {
        \\                value = 30;
        \\            }
        \\        }
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 30), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes comptime while loops" {
    const source_text =
        \\pub fn count() -> u256 {
        \\    return comptime {
        \\        let value = 0;
        \\        while (value < 3) {
        \\            value += 1;
        \\        }
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 3), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes comptime integer for loops" {
    const source_text =
        \\pub fn sum() -> u256 {
        \\    return comptime {
        \\        let total = 0;
        \\        for (4) |i| {
        \\            total += i;
        \\        }
        \\        total;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 6), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes comptime array for loops" {
    const source_text =
        \\pub fn sum() -> u256 {
        \\    return comptime {
        \\        let total = 0;
        \\        for ([1, 2, 3]) |value| {
        \\            total += value;
        \\        }
        \\        total;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 6), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes comptime tuple for loops with index" {
    const source_text =
        \\pub fn sum() -> u256 {
        \\    return comptime {
        \\        let total = 0;
        \\        for ((1, 2, 3)) |value, index| {
        \\            total += value + index;
        \\        }
        \\        total;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 9), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval reads comptime array elements by index" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let values = [7, 8, 9];
        \\        values[1];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 8), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval reads comptime tuple elements by index" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let values = (4, 5, 6);
        \\        values[2];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 6), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval mutates comptime array elements by index" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let values = [7, 8, 9];
        \\        values[1] = 42;
        \\        values[1];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 42), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval applies indexed compound assignment on arrays" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let values = [7, 8, 9];
        \\        values[1] += 4;
        \\        values[1];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 12), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval reads direct struct literal fields" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: bool;
        \\}
        \\
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        Pair { first: 7, second: false }.first;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 7), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval reads bound struct fields" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: bool;
        \\}
        \\
        \\pub fn get() -> bool {
        \\    return comptime {
        \\        let pair = Pair { first: 7, second: true };
        \\        pair.second;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval mutates bound struct fields" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: bool;
        \\}
        \\
        \\pub fn get() -> bool {
        \\    return comptime {
        \\        let pair = Pair { first: 7, second: false };
        \\        pair.second = true;
        \\        pair.second;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval applies compound assignment to struct fields" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: bool;
        \\}
        \\
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let pair = Pair { first: 7, second: false };
        \\        pair.first += 5;
        \\        pair.first;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 12), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval reads string length and indexing" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let text = "hello";
        \\        text.length + text[1];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 106), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval reads bytes length and indexing" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let payload = hex"deadbeef";
        \\        payload.len + payload[0];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 226), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval compares enum constants" {
    const source_text =
        \\enum Mode {
        \\    off,
        \\    on,
        \\}
        \\
        \\pub fn same() -> bool {
        \\    return comptime {
        \\        Mode.on == Mode.on;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval compares address literals" {
    const source_text =
        \\pub fn same() -> bool {
        \\    return comptime {
        \\        0x1234567890abcdef1234567890abcdef12345678 == 0x1234567890abcdef1234567890abcdef12345678;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval executes comptime break in while loops" {
    const source_text =
        \\pub fn count() -> u256 {
        \\    return comptime {
        \\        let value = 0;
        \\        while (true) {
        \\            value += 1;
        \\            if (value == 3) {
        \\                break;
        \\            }
        \\        }
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 3), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes comptime continue in for loops" {
    const source_text =
        \\pub fn sum() -> u256 {
        \\    return comptime {
        \\        let total = 0;
        \\        for (5) |i| {
        \\            if (i == 2) {
        \\                continue;
        \\            }
        \\            total += i;
        \\        }
        \\        total;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 8), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval constructs and indexes comptime slices" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let values = @cast(slice[u256], [7, 8, 9]);
        \\        values[1];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 8), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval mutates comptime slices and reads len" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let values = @cast(slice[u256], [7, 8, 9]);
        \\        values[1] += 4;
        \\        values.len;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 3), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval constructs and indexes comptime maps" {
    const source_text =
        \\pub fn get() -> bool {
        \\    return comptime {
        \\        let table = @cast(map<u256, bool>, [(1, false), (2, true)]);
        \\        table[2];
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval mutates comptime maps and reads len" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let table = @cast(map<u256, u256>, [(1, 7)]);
        \\        table[1] += 5;
        \\        table[3] = 9;
        \\        table.length;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 2), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes direct function calls" {
    const source_text =
        \\fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        add(2, 3);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes nested direct function calls" {
    const source_text =
        \\fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\fn bump(v: u256) -> u256 {
        \\    return add(v, 1);
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        bump(add(2, 3));
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 6), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval binds function call arguments by parameter pattern" {
    const source_text =
        \\fn pick(pair: (u256, u256)) -> u256 {
        \\    return pair[1];
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        pick((4, 9));
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 9), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes grouped direct function calls" {
    const source_text =
        \\fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        (add)(2, 3);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval recursion limit leaves recursive calls unresolved" {
    const source_text =
        \\fn loop(v: u256) -> u256 {
        \\    return loop(v);
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        loop(1);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(?compiler.sema.ConstValue, null), consteval.values[ret_stmt.value.?.index()]);
}

test "compiler const eval rejects runtime-only statements in called functions" {
    const source_text =
        \\log Ping(value: u256);
        \\
        \\fn noisy(next: u256) -> u256 {
        \\    log Ping(next);
        \\    return next;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        noisy(7);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(?compiler.sema.ConstValue, null), consteval.values[ret_stmt.value.?.index()]);
}

test "compiler const eval executes comptime associated trait methods" {
    const source_text =
        \\trait Selector {
        \\    comptime fn selector() -> u256;
        \\}
        \\
        \\struct Box {}
        \\
        \\impl Selector for Box {
        \\    comptime fn selector() -> u256 {
        \\        return 7;
        \\    }
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        Box.selector();
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[3]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 7), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval executes comptime receiver trait methods" {
    const source_text =
        \\trait Marker {
        \\    comptime fn marked(self) -> u256;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    comptime fn marked(self) -> u256 {
        \\        return self.value + 1;
        \\    }
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        Box { value: 4 }.marked();
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[3]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval supports sizeOf on concrete and generic types" {
    const source_text =
        \\comptime fn width(comptime T: type) -> u256 {
        \\    return @sizeOf(T);
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        width(address) + @sizeOf([u8; 4]);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 24), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval supports typeName and string concat for ABI signatures" {
    const source_text =
        \\pub fn run() -> string {
        \\    return comptime {
        \\        "transfer(" + @typeName(address) + "," + @typeName(u256) + ")";
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(std.mem.eql(u8, "transfer(address,uint256)", consteval.values[ret_stmt.value.?.index()].?.string));
}

test "compiler const eval supports keccak256 for ABI selector hashes" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        @keccak256("transfer(" + @typeName(address) + "," + @typeName(u256) + ")");
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("transfer(address,uint256)", &hash, .{});
    var hex: [64]u8 = undefined;
    for (hash, 0..) |byte, index| {
        hex[index * 2] = std.fmt.hex_charset[byte >> 4];
        hex[index * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
    }
    var expected = try std.math.big.int.Managed.init(testing.allocator);
    defer expected.deinit();
    try expected.setString(16, hex[0..]);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[ret_stmt.value.?.index()].?.integer.eql(expected));
}

test "compiler const eval compares type values in generic comptime logic" {
    const source_text =
        \\comptime fn is_word(comptime T: type) -> bool {
        \\    return T == u256;
        \\}
        \\
        \\pub fn run() -> bool {
        \\    return comptime {
        \\        is_word(u256);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler surfaces comptime stage diagnostics through db and typecheck" {
    const source_text =
        \\log Ping(value: u256);
        \\
        \\fn noisy(next: u256) -> u256 {
        \\    log Ping(next);
        \\    return next;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        noisy(7);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const consteval_diags = try compilation.db.constEvalDiagnostics(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), consteval_diags.len());
    try testing.expect(std.mem.containsAtLeast(u8, consteval_diags.items.items[0].message, 1, "runtime-only operation in comptime context"));
    try testing.expect(std.mem.containsAtLeast(u8, consteval_diags.items.items[0].message, 1, "noisy"));

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const typecheck_diags = &module_typecheck.diagnostics;
    try testing.expect(typecheck_diags.len() >= 1);
    try testing.expect(std.mem.containsAtLeast(u8, typecheck_diags.items.items[0].message, 1, "runtime-only operation in comptime context"));
}

test "compiler reports missing top-level comptime values through diagnostics" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let value = missingName;
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const consteval_diags = try compilation.db.constEvalDiagnostics(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(consteval_diags, "comptime block did not produce a value"));
}

test "compiler reports missing comptime call values through diagnostics" {
    const source_text =
        \\comptime fn broken() -> u256 {
        \\    let value = missingName;
        \\    value;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        broken();
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const consteval_diags = try compilation.db.constEvalDiagnostics(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(consteval_diags, "comptime call did not produce a value"));
    try testing.expect(diagnosticMessagesContain(consteval_diags, "broken"));
}

test "compiler const eval executes unary returns through comptime call fallback" {
    const source_text =
        \\comptime fn invert(flag: bool) -> bool {
        \\    return !flag;
        \\}
        \\
        \\pub fn run() -> bool {
        \\    return comptime {
        \\        invert(false);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval executes switch returns through comptime call fallback" {
    const source_text =
        \\comptime fn choose(flag: bool) -> u256 {
        \\    return switch (flag) {
        \\        true => 7,
        \\        else => 9,
        \\    };
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        choose(true);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 7), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler rejects unsupported nested comptime expressions" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        comptime {
        \\            7;
        \\        };
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const syntax_diags = try compilation.db.syntaxDiagnostics(compilation.db.sources.module(compilation.root_module_id).file_id);
    try testing.expect(diagnosticMessagesContain(syntax_diags, "expected expression"));
}

test "compiler lowers ghost items into ghost AST nodes" {
    const source_text =
        \\contract Spec {
        \\    ghost const LIMIT: u256 = 1;
        \\    ghost storage var hidden: u256;
        \\    ghost fn helper() -> u256 {
        \\        return LIMIT;
        \\    }
        \\    ghost {
        \\        let shadow = hidden;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    try testing.expectEqual(@as(usize, 4), contract.members.len);

    const ghost_const = ast_file.item(contract.members[0]).Constant;
    try testing.expect(ghost_const.is_ghost);

    const ghost_field = ast_file.item(contract.members[1]).Field;
    try testing.expect(ghost_field.is_ghost);

    const ghost_fn = ast_file.item(contract.members[2]).Function;
    try testing.expect(ghost_fn.is_ghost);

    const ghost_block = ast_file.item(contract.members[3]).GhostBlock;
    const ghost_body = ast_file.body(ghost_block.body);
    try testing.expectEqual(@as(usize, 1), ghost_body.statements.len);
    try testing.expect(ast_file.statement(ghost_body.statements[0]).* == .VariableDecl);
}

test "compiler lowers ghost declarations into verification HIR" {
    const source_text =
        \\contract Spec {
        \\    ghost const LIMIT: u256 = 1;
        \\    ghost storage var hidden: u256;
        \\    ghost fn helper() -> u256 {
        \\        assert(true, "fn");
        \\        return LIMIT;
        \\    }
        \\    ghost {
        \\        assert(hidden == LIMIT, "block");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ghost = true"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.formal = true"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_function"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_variable"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_constant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_assertion"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
}

test "compiler lowers real HIR if and try regions" {
    const source_text =
        \\log Ping(value: u256);
        \\
        \\pub fn flow(ok: bool, next: u256) -> u256 {
        \\    let value = next;
        \\    if (ok) {
        \\        log Ping(next);
        \\    } else {
        \\        assume(next >= 0);
        \\    }
        \\    try {
        \\        assert(ok);
        \\    } catch (err) {
        \\        havoc err;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.log"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assume"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.havoc"));
}

test "compiler lowers real HIR if regions with carried locals" {
    const source_text =
        \\pub fn choose(ok: bool, start: u256) -> u256 {
        \\    let value = start;
        \\    if (ok) {
        \\        value = value + 1;
        \\    } else {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler lowers if statements with early return without placeholders" {
    const source_text =
        \\pub fn choose(ok: bool, start: u256) -> u256 {
        \\    if (ok) {
        \\        return start;
        \\    } else {
        \\        assert(start >= 0);
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.conditional_return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler lowers if statements with carried locals and early return without placeholders" {
    const source_text =
        \\pub fn choose(ok: bool, start: u256) -> u256 {
        \\    let value = start;
        \\    if (ok) {
        \\        return value;
        \\    } else {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.conditional_return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler skips unknown carried locals in if lowering" {
    const source_text =
        \\pub fn choose(flag: bool, value: u256) -> u256 {
        \\    let total = 0;
        \\    let bad = value.missing;
        \\    if (flag) {
        \\        bad = value.missing;
        \\        total = 1;
        \\    } else {
        \\        total = 2;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler lowers try statements with early return without placeholders" {
    const source_text =
        \\pub fn recover(ok: bool, start: u256) -> u256 {
        \\    let value = start;
        \\    try {
        \\        if (ok) {
        \\            return value;
        \\        }
        \\        value = value + 1;
        \\    } catch (err) {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.conditional_return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
}

test "compiler lowers real HIR try regions with carried locals" {
    const source_text =
        \\pub fn recover(start: u256) -> u256 {
        \\    let value = start;
        \\    try {
        \\        value = value + 1;
        \\    } catch (err) {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
}

test "compiler lowers real HIR try regions with shadowed catch locals" {
    const source_text =
        \\pub fn recover(start: u256) -> u256 {
        \\    let err = start;
        \\    try {
        \\        err = err + 1;
        \\    } catch (err) {
        \\        assert(err >= 0);
        \\    }
        \\    return err;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
}

test "compiler parses statement-form try expressions" {
    const source_text =
        \\error E1;
        \\pub fn mayFail(v: u256) -> !void | E1 {
        \\    if (v == 0) {
        \\        return E1;
        \\    }
        \\}
        \\
        \\pub fn run(v: u256) -> bool {
        \\    try {
        \\        try mayFail(v);
        \\        return true;
        \\    } catch (e) {
        \\        return false;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler propagates try expressions inside try statements without default placeholders" {
    const source_text =
        \\struct Pair {
        \\    value: u256;
        \\}
        \\
        \\error Missing;
        \\
        \\pub fn load(ok: bool) -> !Pair | Missing {
        \\    if (!ok) {
        \\        return Missing;
        \\    }
        \\    return Pair { value: 7 };
        \\}
        \\
        \\pub fn run(ok: bool) -> u256 {
        \\    try {
        \\        let pair: Pair = try load(ok);
        \\        return pair.value;
        \\    } catch (e) {
        \\        return 0;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.unwrap"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.default_value"));
}

test "compiler lowers real HIR switch regions with carried locals" {
    const source_text =
        \\pub fn choose(tag: u256, start: u256) -> u256 {
        \\    let value = start;
        \\    switch (tag) {
        \\        0 => {
        \\            value = value + 1;
        \\        },
        \\        else => {
        \\            value = value + 2;
        \\        }
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler lowers switch statements with early return without placeholders" {
    const source_text =
        \\pub fn choose(tag: u256, start: u256) -> u256 {
        \\    let value = start;
        \\    switch (tag) {
        \\        0 => {
        \\            return value;
        \\        },
        \\        else => {
        \\            value = value + 2;
        \\        }
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler lowers labeled switch continue through a real loop" {
    const source_text =
        \\pub fn run(initial: u256) -> u256 {
        \\    var tag = initial;
        \\    again: switch (tag) {
        \\        0 => {
        \\            tag = 1;
        \\            continue :again tag;
        \\        },
        \\        1 => {
        \\            break :again;
        \\        },
        \\        else => {
        \\            break :again;
        \\        }
        \\    }
        \\    return tag;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.store"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler skips unknown carried locals in switch lowering" {
    const source_text =
        \\pub fn choose(flag: bool, value: u256) -> u256 {
        \\    let total = 0;
        \\    let bad = value.missing;
        \\    switch (flag) {
        \\        true => {
        \\            bad = value.missing;
        \\            total = total + 1;
        \\        },
        \\        else => {
        \\            total = total + 2;
        \\        }
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler lowers switch arms with nested for carried locals" {
    const source_text =
        \\pub fn choose(flag: bool, values: slice[u256]) -> u256 {
        \\    let total = 0;
        \\    switch (flag) {
        \\        true => {
        \\            for (values) |value| {
        \\                total = total + value;
        \\            }
        \\        },
        \\        else => {
        \\            total = total + 1;
        \\        }
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers for loops with nested switch carried locals" {
    const source_text =
        \\pub fn sum(values: slice[u256], step: u256) -> u256 {
        \\    let total = 0;
        \\    for (values) |value| {
        \\        switch (step) {
        \\            1 => {
        \\                total = total + value;
        \\            },
        \\            else => {
        \\                total = total + 1;
        \\            }
        \\        }
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler lowers real HIR switch expressions" {
    const source_text =
        \\pub fn choose(tag: u256, start: u256) -> u256 {
        \\    return switch (tag) {
        \\        0 => start + 1,
        \\        else => start + 2,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.switch_expr\""));
}

test "compiler lowers switch expressions with computed constant patterns" {
    const source_text =
        \\pub fn choose(tag: u256) -> u256 {
        \\    return switch (tag) {
        \\        1 + 2 => 7,
        \\        else => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "case 3 =>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.switch_expr\""));
}

test "compiler lowers real HIR switch regions" {
    const source_text =
        \\pub fn classify(v: u256, fallback: u256) -> u256 {
        \\    switch (v) {
        \\        0 => {
        \\            assert(true);
        \\        },
        \\        1...2 => {
        \\            assume(v >= 1);
        \\        },
        \\        else => {
        \\            havoc fallback;
        \\        }
        \\    }
        \\    return v;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assume"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.havoc"));
}

test "compiler lowers switch arms with nested loop breaks without placeholders" {
    const source_text =
        \\pub fn classify(flag: bool, seed: u256) -> u256 {
        \\    let value = seed;
        \\    switch (flag) {
        \\        true => {
        \\            while (value < seed + 1) {
        \\                break;
        \\            }
        \\            value = value + 1;
        \\        },
        \\        else => {
        \\            value = value + 2;
        \\        }
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler lowers real HIR while loops for storage-driven loops" {
    const source_text =
        \\storage count: u256;
        \\
        \\pub fn drain(limit: u256) {
        \\    while (count < limit) {
        \\        count = count + 1;
        \\        assert(count >= 1);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.condition"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sload"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers address-typed storage reads with address result types" {
    const source_text =
        \\contract C {
        \\    storage var address_value: address;
        \\
        \\    pub fn read() -> address {
        \\        return address_value;
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.global \"address_value\" : !ora.address"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sload \"address_value\" : !ora.address"));
}

test "compiler lowers enum-typed storage reads with lowered integer result types" {
    const source_text =
        \\enum Status { A, B }
        \\
        \\contract C {
        \\    storage var status: Status;
        \\
        \\    pub fn read() -> Status {
        \\        return status;
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.global \"status\" : i256"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sload \"status\" : i256"));
}

test "compiler skips unknown carried locals in while lowering" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let total = 0;
        \\    let bad = total.missing;
        \\    while (total < limit) {
        \\        bad = total.missing;
        \\        total = total + 1;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with carried locals" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        value = value + 1;
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with top-level break and continue" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        value = value + 1;
        \\        continue;
        \\    }
        \\    let seen = 0;
        \\    while (seen < limit) {
        \\        seen = seen + 1;
        \\        break;
        \\    }
        \\    return value + seen;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with branch-local break and continue" {
    const source_text =
        \\pub fn count(limit: u256, stop_now: bool) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        if (stop_now) {
        \\            break;
        \\        } else {
        \\            value = value + 1;
        \\            continue;
        \\        }
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with nested inner-loop control" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        let step = 0;
        \\        while (step < 1) {
        \\            step = step + 1;
        \\            continue;
        \\        }
        \\        while (value < limit) {
        \\            break;
        \\        }
        \\        value = value + 1;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with nested merged if locals" {
    const source_text =
        \\pub fn count(limit: u256, take_big_step: bool) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        if (take_big_step) {
        \\            value = value + 2;
        \\        } else {
        \\            value = value + 1;
        \\        }
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with nested merged try locals" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        try {
        \\            value = value + 2;
        \\        } catch (err) {
        \\            value = value + 1;
        \\        }
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with nested merged switch locals" {
    const source_text =
        \\pub fn count(limit: u256, step: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        switch (step) {
        \\            1 => {
        \\                value = value + 1;
        \\            },
        \\            else => {
        \\                value = value + 2;
        \\            }
        \\        }
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers deeply nested carried locals without placeholders" {
    const source_text =
        \\pub fn count(limit: u256, flag: bool, step: u256, start: u256) -> u256 {
        \\    let value = start;
        \\    while (value < limit) {
        \\        if (flag) {
        \\            switch (step) {
        \\                0 => {
        \\                    try {
        \\                        value = value + 1;
        \\                    } catch (err) {
        \\                        value = value + 2;
        \\                    }
        \\                },
        \\                else => {
        \\                    try {
        \\                        value = value + 3;
        \\                    } catch (err) {
        \\                        value = value + 4;
        \\                    }
        \\                }
        \\            }
        \\        } else {
        \\            value = value + 5;
        \\        }
        \\        assert(value >= start);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
}

test "compiler lowers real HIR while loops with nested switch expressions" {
    const source_text =
        \\pub fn count(limit: u256, step: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        value = switch (step) {
        \\            1 => value + 1,
        \\            else => value + 2,
        \\        };
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers while loops with early return without placeholders" {
    const source_text =
        \\pub fn count(limit: u256, stop_at: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        if (value == stop_at) {
        \\            return value;
        \\        }
        \\        value = value + 1;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.conditional_return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers while loops with typed carried locals without explicit initializer" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value: u256;
        \\    let i = 0;
        \\    while (i < limit) {
        \\        value = value + 1;
        \\        i = i + 1;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler tracks storage_class on variable declarations" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() {
        \\    storage var x: u256 = 0;
        \\    memory var y: u256 = 0;
        \\    let z: u256 = 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);

    const decl_x = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.storage, typecheck.pattern_types[decl_x.pattern.index()].region);

    const decl_y = ast_file.statement(body.statements[1]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_y.pattern.index()].region);

    const decl_z = ast_file.statement(body.statements[2]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_z.pattern.index()].region);
}

test "compiler rejects region-incompatible variable initialization" {
    const source_text =
        \\contract Vault {
        \\    tstore var pending: u256;
        \\}
        \\
        \\pub fn example() {
        \\    storage var x: u256 = pending;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(!diags.isEmpty());
    try testing.expect(diagnosticMessagesContain(diags, "declaration expects region 'storage', found 'transient'"));
}

test "compiler rejects region-incompatible field initializer" {
    const source_text =
        \\contract Vault {
        \\    tstore var pending: u256;
        \\    storage var committed: u256 = pending;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(!diags.isEmpty());
    try testing.expect(diagnosticMessagesContain(diags, "field 'committed' expects region"));
}

test "compiler allows passing storage values to function parameters" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn helper(x: u256) -> u256 {
        \\    return x;
        \\}
        \\
        \\pub fn example() -> u256 {
        \\    return helper(total);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler rejects type mismatch in function arguments" {
    const source_text =
        \\pub fn helper(x: u256) -> u256 {
        \\    return x;
        \\}
        \\
        \\pub fn example() -> u256 {
        \\    return helper(true);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(!diags.isEmpty());
    try testing.expect(diagnosticMessagesContain(diags, "expected argument type"));
}

test "compiler allows returning storage values from functions" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() -> u256 {
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler rejects return type mismatch" {
    const source_text =
        \\pub fn example() -> u256 {
        \\    return true;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const diags = &module_typecheck.diagnostics;
    try testing.expect(!diags.isEmpty());
    try testing.expect(diagnosticMessagesContain(diags, "return expects type"));
}

test "compiler tracks storage_class on inferred-type variable declarations" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() {
        \\    storage var x = 0;
        \\    memory var y = 0;
        \\    tstore var z = 0;
        \\    let w = 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);

    const decl_x = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.storage, typecheck.pattern_types[decl_x.pattern.index()].region);

    const decl_y = ast_file.statement(body.statements[1]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_y.pattern.index()].region);

    const decl_z = ast_file.statement(body.statements[2]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.transient, typecheck.pattern_types[decl_z.pattern.index()].region);

    const decl_w = ast_file.statement(body.statements[3]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.memory, typecheck.pattern_types[decl_w.pattern.index()].region);
}

test "compiler tracks tstore var inside function body" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn example() {
        \\    tstore var temp: u256 = 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);

    const decl = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.Region.transient, typecheck.pattern_types[decl.pattern.index()].region);
}

test "compiler allows passing calldata values to function parameters" {
    const source_text =
        \\pub fn helper(x: u256) -> u256 {
        \\    return x;
        \\}
        \\
        \\pub fn example(value: u256) -> u256 {
        \\    return helper(value);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler emits ABI attrs for public contract entries" {
    const source_text =
        \\contract Entry {
        \\    pub fn init(owner: address) {}
        \\
        \\    pub fn run(owner: address, amount: u256) -> bool {
        \\        return owner == owner && amount > 0;
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.selector"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_params"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"address\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"uint256\""));
}

test "compiler rejects invalid constructor shape" {
    const source_text =
        \\contract Entry {
        \\    pub fn init(self, owner: address) -> bool {
        \\        return true;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(!module_typecheck.diagnostics.isEmpty());
}

test "compiler emits error selector and error-union ABI attrs" {
    const source_text =
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract Vault {
        \\    pub fn withdraw(amount: u256) -> !u256 | InsufficientBalance {
        \\        return error InsufficientBalance(amount, amount);
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.error_selector"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"0xcf479181\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.returns_error_union"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"uint256\""));
}

test "compiler emits tuple ABI return attrs for public error unions" {
    const source_text =
        \\error Failure();
        \\
        \\contract Vault {
        \\    pub fn quote() -> !(u256, bool) | Failure {
        \\        return (1, true);
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.returns_error_union"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"tuple\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return_words"));
}

test "compiler emits struct ABI return attrs for public error unions" {
    const source_text =
        \\struct Snapshot {
        \\    owner: address;
        \\    amount: u256;
        \\}
        \\
        \\error Failure();
        \\
        \\contract Vault {
        \\    pub fn snapshot() -> !Snapshot | Failure {
        \\        return Snapshot { owner: 0x0000000000000000000000000000000000000000, amount: 1 };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.returns_error_union"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"tuple\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return_words"));
}

test "compiler emits recursive ABI layout attrs for dynamic aggregate public returns" {
    const source_text =
        \\struct Snapshot {
        \\    owner: address;
        \\    note: string;
        \\}
        \\
        \\error Failure();
        \\
        \\contract Vault {
        \\    pub fn quote() -> !(u256, string) | Failure {
        \\        return (1, "ok");
        \\    }
        \\
        \\    pub fn snapshot() -> !Snapshot | Failure {
        \\        return Snapshot { owner: 0x0000000000000000000000000000000000000000, note: "hi" };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return_layout"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(uint256,string)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(address,string)\""));
}

test "compiler preserves error selectors through OraToSIR" {
    const source_text =
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract Vault {
        \\    pub fn withdraw(amount: u256) -> !u256 | InsufficientBalance {
        \\        return error InsufficientBalance(amount, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.error_selectors"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"0xcf479181\""));
}

test "compiler lowers payload error return constructors through OraToSIR" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn run() -> !bool | Failure {
        \\        return error Failure(7);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.return"));
}

test "compiler supports call-style payload error constructors" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn run(flag: bool) -> !bool | Failure {
        \\        if (flag) {
        \\            return Failure(7);
        \\        }
        \\        return false;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"Failure\""));
}

test "compiler lowers payload-bearing narrow success error unions through OraToSIR" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn run(flag: bool) -> !bool | Failure {
        \\        return flag;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.ok"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.err"));
}

test "compiler carries payload-bearing narrow error unions across function calls through OraToSIR" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\fn helper(flag: bool) -> !bool | Failure {
        \\    return flag;
        \\}
        \\
        \\contract Probe {
        \\    pub fn run(flag: bool) -> !bool | Failure {
        \\        return helper(flag);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.icall"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.ok"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.err"));
}

test "compiler lowers try on payload-bearing narrow error unions through OraToSIR" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\fn helper(flag: bool) -> !bool | Failure {
        \\    return flag;
        \\}
        \\
        \\contract Probe {
        \\    pub fn run(flag: bool) -> !bool | Failure {
        \\        return try helper(flag);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.yield"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.conditional_return"));
}

test "compiler lowers bare assert to runtime revert through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(flag: bool) {
        \\        assert(flag);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.invalid"));
}

test "compiler lowers message assert to runtime revert payload through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(flag: bool) {
        \\        assert(flag, "bad");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.store8"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.assert"));
}

test "dispatcher translates public zero-payload error unions to ABI reverts" {
    const source_text =
        \\error Failure();
        \\
        \\contract Check {
        \\    pub fn run() -> !u256 | Failure {
        \\        return Failure();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @main"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.error_selectors"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.return"));
}

test "dispatcher translates public payload error unions to ABI reverts" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Check {
        \\    pub fn run() -> !u256 | Failure {
        \\        return Failure(7);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @main"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.addptr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "sir.store"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.error_selectors"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.return"));
}

test "dispatcher translates public tuple-success error unions" {
    const source_text =
        \\extern trait View {
        \\    staticcall fn quote(self) -> (u256, bool);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Check {
        \\    storage var target: address;
        \\
        \\    pub fn run() -> !(u256, bool) | ExternalCallFailed {
        \\        return external<View>(target, gas: 50000).quote();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @main"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.return"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "public error-union dispatcher currently supports only scalar ABI success payloads"));
}

test "dispatcher translates public struct-success error unions" {
    const source_text =
        \\struct Snapshot {
        \\    owner: address;
        \\    amount: u256;
        \\}
        \\
        \\extern trait View {
        \\    staticcall fn snapshot(self) -> Snapshot;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Check {
        \\    storage var target: address;
        \\
        \\    pub fn run() -> !Snapshot | ExternalCallFailed {
        \\        return external<View>(target, gas: 50000).snapshot();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @main"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.return"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "public error-union dispatcher currently supports scalar, static tuple/struct, bytes/string, and static-base dynamic array ABI success payloads"));
}

test "dispatcher translates public bytes-success error unions" {
    const source_text =
        \\extern trait View {
        \\    staticcall fn blob(self) -> bytes;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Check {
        \\    storage var target: address;
        \\
        \\    pub fn run() -> !bytes | ExternalCallFailed {
        \\        return external<View>(target, gas: 50000).blob();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @main"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.add"));
}

test "dispatcher translates public string-success error unions" {
    const source_text =
        \\extern trait View {
        \\    staticcall fn name(self) -> string;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Check {
        \\    storage var target: address;
        \\
        \\    pub fn run() -> !string | ExternalCallFailed {
        \\        return external<View>(target, gas: 50000).name();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @main"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.add"));
}

test "dispatcher translates public dynamic array success error unions" {
    const source_text =
        \\extern trait View {
        \\    staticcall fn values(self) -> slice[u256];
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Check {
        \\    storage var target: address;
        \\
        \\    pub fn run() -> !slice[u256] | ExternalCallFailed {
        \\        return external<View>(target, gas: 50000).values();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"uint256[]\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.mul"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.add"));
}

test "dispatcher translates public dynamic tuple success error unions" {
    const source_text =
        \\extern trait View {
        \\    staticcall fn quote(self) -> (u256, string);
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Check {
        \\    storage var target: address;
        \\
        \\    pub fn run() -> !(u256, string) | ExternalCallFailed {
        \\        return external<View>(target, gas: 50000).quote();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(uint256,string)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.addptr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "public error-union dispatcher currently supports scalar, static tuple/struct, bytes/string, and static-base dynamic array ABI success payloads"));
}

test "dispatcher translates public dynamic struct success error unions" {
    const source_text =
        \\struct Snapshot {
        \\    owner: address;
        \\    note: string;
        \\}
        \\
        \\extern trait View {
        \\    staticcall fn snapshot(self) -> Snapshot;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Check {
        \\    storage var target: address;
        \\
        \\    pub fn run() -> !Snapshot | ExternalCallFailed {
        \\        return external<View>(target, gas: 50000).snapshot();
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(address,string)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.addptr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "public error-union dispatcher currently supports scalar, static tuple/struct, bytes/string, and static-base dynamic array ABI success payloads"));
}

test "dispatcher translates payload-bearing extern trait errors to ABI reverts" {
    const source_text =
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool errors(InsufficientBalance);
        \\}
        \\
        \\error ExternalCallFailed;
        \\error InsufficientBalance(required: u256, available: u256);
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn send(to: address, amount: u256) -> !bool | ExternalCallFailed | InsufficientBalance {
        \\        return external<ERC20>(token, gas: 50000).transfer(to, amount);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @main"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.call"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.error_selectors"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
}

test "ora dialect exposes external call ops through C API" {
    const ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(ctx);
    try testing.expect(mlir.oraDialectRegister(ctx));

    const loc = mlir.oraLocationUnknownGet(ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    defer mlir.oraModuleDestroy(module);
    const body = mlir.oraModuleGetBody(module);

    const i32_ty = mlir.oraIntegerTypeCreate(ctx, 32);
    const i1_ty = mlir.oraBoolTypeGet(ctx);
    const i256_ty = mlir.oraIntegerTypeGet(ctx, 256, false);
    const addr_ty = mlir.oraAddressTypeGet(ctx);

    const selector_attr = mlir.oraIntegerAttrCreateI64FromType(i32_ty, 0xa9059cbb);
    const arg_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(ctx, mlir.oraStringRefCreateFromCString("address")),
        mlir.oraStringAttrCreate(ctx, mlir.oraStringRefCreateFromCString("uint256")),
    };
    const arg_types_attr = mlir.oraArrayAttrCreate(ctx, arg_type_attrs.len, &arg_type_attrs);
    const return_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(ctx, mlir.oraStringRefCreateFromCString("bool")),
    };
    const return_types_attr = mlir.oraArrayAttrCreate(ctx, return_type_attrs.len, &return_type_attrs);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const gas_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 50000);
    const amount_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);

    const target_const = mlir.oraArithConstantOpCreate(ctx, loc, addr_ty, zero_attr);
    const gas_const = mlir.oraArithConstantOpCreate(ctx, loc, i256_ty, gas_attr);
    const amount_const = mlir.oraArithConstantOpCreate(ctx, loc, i256_ty, amount_attr);
    mlir.oraBlockAppendOwnedOperation(body, target_const);
    mlir.oraBlockAppendOwnedOperation(body, gas_const);
    mlir.oraBlockAppendOwnedOperation(body, amount_const);

    const target = mlir.oraOperationGetResult(target_const, 0);
    const gas = mlir.oraOperationGetResult(gas_const, 0);
    const amount = mlir.oraOperationGetResult(amount_const, 0);

    const encode_operands = [_]mlir.MlirValue{ target, amount };
    const encode_op = mlir.oraAbiEncodeOpCreate(ctx, loc, selector_attr, arg_types_attr, &encode_operands, encode_operands.len, i256_ty);
    mlir.oraBlockAppendOwnedOperation(body, encode_op);

    const calldata = mlir.oraOperationGetResult(encode_op, 0);
    const external_call_op = mlir.oraExternalCallOpCreate(
        ctx,
        loc,
        mlir.oraStringRefCreateFromCString("call"),
        mlir.oraStringRefCreateFromCString("ERC20"),
        mlir.oraStringRefCreateFromCString("transfer"),
        target,
        gas,
        calldata,
        i1_ty,
        i256_ty,
    );
    mlir.oraBlockAppendOwnedOperation(body, external_call_op);

    const returndata = mlir.oraOperationGetResult(external_call_op, 1);
    const decode_op = mlir.oraAbiDecodeOpCreate(ctx, loc, return_types_attr, returndata, i1_ty);
    mlir.oraBlockAppendOwnedOperation(body, decode_op);

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_encode"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"call\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"ERC20\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"transfer\""));
}

test "compiler examples convert through SIR" {
    const example_paths = [_][]const u8{
        "ora-example/smoke.ora",
        "ora-example/no_return_test.ora",
        "ora-example/dce_test.ora",
        "ora-example/statements/contract_declaration.ora",
    };

    for (example_paths) |path| {
        try expectOraToSirConverts(path);
    }
}

test "compiler converts contract storage through explicit slot metadata" {
    const source_text =
        \\contract Vault {
        \\    storage var balance: u256 = 1;
        \\    storage var owner: address;
        \\
        \\    pub fn read() -> u256 {
        \\        return balance;
        \\    }
        \\
        \\    pub fn write(next: u256, who: address) {
        \\        balance = next;
        \\        owner = who;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "slot_balance"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "slot_owner"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.sload"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.sstore"));
}

test "compiler converts narrowed carried locals in nested scf ifs" {
    const source_text =
        \\contract Test {
        \\    pub fn update(current_status: u8, user_borrow: u256, health: u256) -> u8 {
        \\        var new_status: u8 = current_status;
        \\        if (current_status == 0) {
        \\            if (user_borrow == 0) {
        \\                new_status = 1;
        \\            } else {
        \\                if (health < 10000) {
        \\                    new_status = 3;
        \\                }
        \\            }
        \\        }
        \\        return new_status;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));
}

test "compiler examples leave no residual Ora runtime ops after OraToSIR" {
    const example_paths = [_][]const u8{
        "ora-example/smoke.ora",
        "ora-example/no_return_test.ora",
        "ora-example/dce_test.ora",
        "ora-example/statements/contract_declaration.ora",
        "ora-example/apps/erc20.ora",
        "ora-example/apps/counter.ora",
        "ora-example/apps/arithmetic_probe.ora",
        "ora-example/apps/erc20_verified.ora",
        "ora-example/apps/defi_lending_pool.ora",
        "ora-example/apps/erc20_bitfield_comptime_generics.ora",
        "ora-example/array_operations.ora",
        "ora-example/structs/basic_structs.ora",
        "ora-example/tuples/tuple_basics.ora",
        "ora-example/bitfields/basic_bitfield_storage.ora",
        "ora-example/bitfields/bitfield_map_values.ora",
        "ora-example/comptime/comptime_basics.ora",
        "ora-example/comptime/comptime_functions.ora",
        "ora-example/errors/try_catch.ora",
        "ora-example/locks/lock_runtime_map_guard.ora",
        "ora-example/locks/lock_runtime_scalar_guard.ora",
        "ora-example/locks/lock_runtime_independent_roots.ora",
        "ora-example/regions/region_ok_storage_map.ora",
        "ora-example/regions/region_ok_storage_tstore_map_and_scalar_writes.ora",
        "ora-example/regions/region_ok_storage_tstore_same_type_writes.ora",
        "ora-example/refinements/dispatcher_refinement_e2e.ora",
        "ora-example/refinements/basic_refinements.ora",
        "ora-example/refinements/comprehensive_test.ora",
        "ora-example/refinements/guards_showcase.ora",
        "ora-example/smt/verification/state_invariants.ora",
        "ora-example/vault/02_errors.ora",
    };

    for (example_paths) |path| {
        var compilation = try compilePackage(path);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module));

        const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
        defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
        const rendered = module_text_ref.data[0..module_text_ref.length];

        try expectNoResidualOraRuntimeOps(rendered);
    }
}
