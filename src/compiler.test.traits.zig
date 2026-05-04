const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;
const mlir = @import("mlir_c_api").c;
const z3_verification = @import("ora_z3_verification");

const h = @import("compiler.test.helpers.zig");
const compileText = h.compileText;
const renderHirTextForSource = h.renderHirTextForSource;
const renderOraMlirForSource = h.renderOraMlirForSource;
const renderSirTextForModule = h.renderSirTextForModule;
const compilePackage = h.compilePackage;
const expectOraToSirConverts = h.expectOraToSirConverts;
const expectNoResidualOraRuntimeOps = h.expectNoResidualOraRuntimeOps;
const VerificationProbeSummary = h.VerificationProbeSummary;
const expectVerificationProbeEquivalent = h.expectVerificationProbeEquivalent;
const verifyExampleWithoutDegradation = h.verifyExampleWithoutDegradation;
const verifyTextWithoutDegradation = h.verifyTextWithoutDegradation;
const verifyTextWithoutDegradationWithTimeout = h.verifyTextWithoutDegradationWithTimeout;
const firstChildNodeOfKind = h.firstChildNodeOfKind;
const nthChildNodeOfKind = h.nthChildNodeOfKind;
const containsNodeOfKind = h.containsNodeOfKind;
const findVariablePatternByName = h.findVariablePatternByName;
const diagnosticMessagesContain = h.diagnosticMessagesContain;
const countDiagnosticMessages = h.countDiagnosticMessages;
const DiagnosticProbePhase = h.DiagnosticProbePhase;
const expectDiagnosticProbeContains = h.expectDiagnosticProbeContains;
const containsEffectSlot = h.containsEffectSlot;
const containsKeyedEffectSlot = h.containsKeyedEffectSlot;
const nthDescendantNodeOfKind = h.nthDescendantNodeOfKind;
const nthDescendantNodeOfKindInner = h.nthDescendantNodeOfKindInner;

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
    try testing.expectEqual(compiler.ast.ReceiverKind.value_self, trait_item.methods[0].receiver_kind);
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
    try testing.expectEqual(compiler.ast.ReceiverKind.extern_self, trait_item.methods[0].receiver_kind);
    try testing.expectEqual(compiler.ast.ReceiverKind.extern_self, trait_item.methods[1].receiver_kind);
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
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "type 'external<ERC20>' has no field 'missing'"));
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"ExternalCallFailed\""));
}

test "compiler lowers extern trait calls with aggregate and enum parameters to ABI layouts" {
    const source_text =
        \\struct Snapshot {
        \\    owner: address;
        \\    amount: u256;
        \\}
        \\
        \\enum Status : u8 {
        \\    Open,
        \\    Closed,
        \\}
        \\
        \\extern trait Sink {
        \\    staticcall fn submit(self, snapshot: Snapshot, status: Status, quote: (u256, bool)) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var target: address;
        \\
        \\    pub fn probe(owner: address, amount: u256, status: Status) -> !bool | ExternalCallFailed {
        \\        let snapshot: Snapshot = Snapshot{ owner: owner, amount: amount };
        \\        return external<Sink>(target, gas: 50000).submit(snapshot, status, (amount, true));
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.abi_encode"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"(address,uint256)\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"uint8\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"(uint256,bool)\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.external_call"));
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"InsufficientBalance\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"InvalidRecipient\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"ExternalCallFailed\""));
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"ExternalCallFailed\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"InsufficientBalance\""));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
        trait_interface.methods[0].receiver_kind != .none,
        trait_interface.methods[0].param_types,
    );
    defer testing.allocator.free(transfer_signature);
    try testing.expectEqualStrings("transfer(address,uint256)", transfer_signature);

    const balance_signature = try compiler.hir.abi.signatureForMethod(
        testing.allocator,
        trait_interface.methods[1].name,
        trait_interface.methods[1].receiver_kind != .none,
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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @SafeCounter.Counter.get"));
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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @SafeCounter.Counter.get"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));

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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @SafeCounter.Counter.get"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));

    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.parallel = false;

    var result = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
}

test "compiler inherits trait method clauses onto impl methods" {
    const source_text =
        \\trait Echo {
        \\    fn echo(self, amount: u256) -> u256
        \\        requires(amount > 0)
        \\        ensures(result == amount);
        \\}
        \\
        \\contract Counter {}
        \\
        \\impl Echo for Counter {
        \\    fn echo(self, value: u256) -> u256 {
        \\        return value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.requires"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "trait_method_contract"));

    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.parallel = false;

    var result = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
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

test "compiler reports duplicate impls across imported modules" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dependency_source =
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
    ;

    try tmp.dir.writeFile(.{ .sub_path = "a.ora", .data = dependency_source });
    try tmp.dir.writeFile(.{ .sub_path = "b.ora", .data = dependency_source });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const a = @import("./a.ora");
        \\comptime const b = @import("./b.ora");
        \\
        \\fn run() -> bool {
        \\    return true;
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "duplicate impl for trait 'Marker' and type 'Box'"));
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
    try testing.expectEqual(compiler.ast.ReceiverKind.value_self, trait_interface.?.methods[0].receiver_kind);
    try testing.expectEqual(compiler.sema.TypeKind.integer, trait_interface.?.methods[0].return_type.kind());
    try testing.expectEqualStrings("decimals", trait_interface.?.methods[1].name);
    try testing.expectEqual(compiler.ast.ReceiverKind.none, trait_interface.?.methods[1].receiver_kind);

    const impl_interface = typecheck.implInterfaceByNames("ERC20", "Token");
    try testing.expect(impl_interface != null);
    try testing.expectEqual(@as(usize, 2), impl_interface.?.methods.len);
    try testing.expectEqualStrings("totalSupply", impl_interface.?.methods[0].name);
    try testing.expectEqual(compiler.ast.ReceiverKind.value_self, impl_interface.?.methods[0].receiver_kind);
    try testing.expectEqualStrings("decimals", impl_interface.?.methods[1].name);
}

test "compiler parses and lowers trait bounds on generic functions" {
    const source_text =
        \\trait Comparable {
        \\    fn compare(self, other: u256) -> bool;
        \\}
        \\
        \\fn keep(comptime T: type, value: T) -> T where T: Comparable {
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
    try testing.expectEqual(@as(usize, 1), function.trait_bounds.len);
    try testing.expectEqualStrings("T", function.trait_bounds[0].parameter_name);
    try testing.expectEqualStrings("Comparable", function.trait_bounds[0].trait_name);
}

test "compiler rejects duplicate trait bounds" {
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

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "duplicate trait bound 'T: Comparable'"));
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

test "compiler accepts imported bounded generic calls for implemented trait types" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
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
        \\fn keep(comptime T: type, value: T) -> T where T: Marker {
        \\    return value;
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\fn run(value: dep.Box) -> dep.Box {
        \\    return dep.keep(value);
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @dep.keep__Box"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @dep.keep__Box"));
}

test "compiler rejects imported bounded generic calls for unimplemented trait types" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
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
        \\fn keep(comptime T: type, value: T) -> T where T: Marker {
        \\    return value;
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\struct Other {
        \\    value: u256,
        \\}
        \\
        \\fn run(value: Other) -> Other {
        \\    return dep.keep(value);
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "type 'Other' does not implement trait 'Marker'"));
}

test "compiler lowers imported trait-bound generic value method calls" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
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
        \\fn choose(comptime T: type, value: T) -> bool where T: Marker {
        \\    return value.marked();
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\fn run(value: dep.Box) -> bool {
        \\    return dep.choose(value);
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @dep.choose__Box"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Marker.Box.marked"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @Marker.Box.marked"));
}

test "compiler lowers imported trait-bound generic associated method calls" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
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
        \\fn choose(comptime T: type) -> bool where T: Factory {
        \\    return T.make();
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\fn run() -> bool {
        \\    return dep.choose(dep.Box);
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @dep.choose__Box"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Factory.Box.make"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @Factory.Box.make"));
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Marker.Box.marked"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @choose__Box"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @run"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @Marker.Box.marked"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @choose__Box"));
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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Marker.Box.marked__"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @Marker.Box.marked__"));
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

test "compiler lowers same-named impl methods as trait-qualified symbols" {
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
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Left.Box.mark"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Right.Box.mark"));
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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Factory.Box.make"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @choose__Box"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "call @Factory.Box.make"));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
