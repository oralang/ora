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

test "compiler diagnostic release matrix stays readable" {
    try expectDiagnosticProbeContains(
        \\trait Plain {
        \\    fn ping(self) -> bool;
        \\}
        \\
        \\extern trait ERC20 {
        \\    staticcall fn balanceOf(self, owner: address) -> u256;
        \\}
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\
        \\    pub fn badMissingGas(user: address) -> u256 {
        \\        return external<ERC20>(token).balanceOf(user);
        \\    }
        \\}
    ,
        .syntax,
        "expected ', gas: ...' in external proxy",
    );

    try expectDiagnosticProbeContains(
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Nonexistent(7);
        \\}
    ,
        .resolution,
        "undefined name 'Nonexistent'",
    );

    try expectDiagnosticProbeContains(
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Failure(1, 2, 3);
        \\}
    ,
        .typecheck,
        "expected 1 arguments, found 3",
    );

    try expectDiagnosticProbeContains(
        \\extern trait ERC20 {
        \\    call fn transfer(self, to: address, amount: u256) -> bool;
        \\}
        \\
        \\error ExternalCallFailed;
        \\
        \\contract Vault {
        \\    storage var token: address;
        \\    storage var balance: u256;
        \\
        \\    pub fn bad(to: address) {
        \\        balance = 1;
        \\        let call_result = external<ERC20>(token, gas: 50000).transfer(to, balance);
        \\        _ = call_result;
        \\        balance = 2;
        \\    }
        \\}
    ,
        .typecheck,
        "cannot write storage slot 'balance' after external call because it was written before the call",
    );
}

test "compiler rejects payloadless named error arms that bind a payload" {
    const source_text =
        \\error Failure;
        \\error Denied;
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Failure(err) => 7,
        \\        Denied => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "payloadless named error match arms cannot bind a payload; use 'Failure =>' instead"));
}

test "compiler rejects multi-field named error payload bindings" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\error Denied;
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Failure(err) => 7,
        \\        Denied => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "named error payload bindings must match the error payload field count"));
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
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "return expects type '!u256 | ErrorA', found 'ErrorB'"));
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

test "compiler rejects mixed explicit and implicit string enum values" {
    const source_text =
        \\enum ErrorCode : string {
        \\    InvalidInput = "ERR_INVALID_INPUT",
        \\    Legacy,
        \\    Unauthorized = "ERR_UNAUTHORIZED",
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "string enums with explicit values must assign every variant explicitly"));
}

test "compiler rejects bytes enums without explicit values for every variant" {
    const source_text =
        \\enum Signature : bytes {
        \\    A = hex"deadbeef",
        \\    B,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "bytes enums currently require explicit values for every variant"));
}

test "compiler parses enum explicit value expressions without syntax diagnostics" {
    const source_text =
        \\contract T {
        \\    enum TestEnum {
        \\        Value1 = 5,
        \\        Value2 = 10 + 20,
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expectEqual(@as(usize, 0), diags.items.items.len);
}

test "compiler rejects non-comptime enum explicit value expressions" {
    const source_text =
        \\storage var seed: u256;
        \\
        \\enum Bad {
        \\    A = seed + 1,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "explicit values for integer enums must be compile-time integer expressions"));
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
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "declaration expects type '[bool; 2]', found '[integer; 2]'"));
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
    try testing.expectEqual(@as(usize, 4), type_diags.len());
    try testing.expectEqual(@as(usize, 4), countDiagnosticMessages(type_diags, "constant value 256 does not fit in type 'u8'"));
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

test "compiler rejects directly recursive runtime structs" {
    const source_text =
        \\struct Node {
        \\    next: Node,
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
}

test "compiler rejects mutually recursive runtime structs" {
    const source_text =
        \\struct A {
        \\    b: B,
        \\}
        \\
        \\struct B {
        \\    a: A,
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
}

test "compiler rejects product-contained recursive runtime structs" {
    const source_text =
        \\struct Node {
        \\    tuple_path: (u256, Node),
        \\    array_path: [Node; 2],
        \\    anonymous_path: struct { child: Node },
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
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

test "compiler rejects public ABI signatures using payload-carrying enums" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\contract Entry {
        \\    pub fn accept(value: Event) -> u256 {
        \\        return 0;
        \\    }
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "unsupported ABI type");
}

test "compiler rejects directly recursive payload-carrying enums" {
    const source_text =
        \\enum Tree {
        \\    Leaf,
        \\    Node(Tree),
        \\}
    ;

    try expectDiagnosticProbeContains(source_text, .typecheck, "recursive runtime ADTs must use slice or map indirection");
}
