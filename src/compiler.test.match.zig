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

fn diagnosticContains(
    diags: *const compiler.diagnostics.DiagnosticList,
    severity: compiler.diagnostics.Severity,
    needle: []const u8,
) bool {
    for (diags.items.items) |diag| {
        if (diag.severity == severity and std.mem.indexOf(u8, diag.message, needle) != null) return true;
    }
    return false;
}

fn diagnosticSeverityCount(
    diags: *const compiler.diagnostics.DiagnosticList,
    severity: compiler.diagnostics.Severity,
) usize {
    var count: usize = 0;
    for (diags.items.items) |diag| {
        if (diag.severity == severity) count += 1;
    }
    return count;
}

test "compiler syntax parses match statements as match syntax nodes" {
    const source_text =
        \\pub fn run(value: u256) -> u256 {
        \\    match (value) {
        \\        0 => return 0;
        \\        else => {
        \\            return value;
        \\        }
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
    const match_stmt = nthChildNodeOfKind(body.?, .MatchStmt, 0);
    try testing.expect(match_stmt != null);
    try testing.expect(nthChildNodeOfKind(body.?, .SwitchStmt, 0) == null);
    try testing.expect(nthChildNodeOfKind(match_stmt.?, .SwitchArm, 0) != null);
    try testing.expect(nthChildNodeOfKind(match_stmt.?, .SwitchArm, 1) != null);
}

test "compiler lowers match expressions through existing switch expression path" {
    const source_text =
        \\pub fn run(flag: bool) -> u256 {
        \\    let value = match (flag) {
        \\        true => 1,
        \\        else => 0,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);
    try testing.expect(
        std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr") or
            std.mem.containsAtLeast(u8, hir_text, 1, "ora.match_expr") or
            std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"),
    );
}

test "compiler syntax parses match constructor-style arm patterns" {
    const source_text =
        \\error Failure;
        \\pub fn run(value: Result<u256, Failure>) -> u256 {
        \\    match (value) {
        \\        Ok(inner) => return inner;
        \\        Err(err) => return 0;
        \\        else => return 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = firstChildNodeOfKind(root, .FunctionItem).?;
    const body = firstChildNodeOfKind(function, .Body).?;
    const match_stmt = nthChildNodeOfKind(body, .MatchStmt, 0).?;
    _ = nthChildNodeOfKind(match_stmt, .SwitchArm, 0).?;
    _ = nthChildNodeOfKind(match_stmt, .SwitchArm, 1).?;
}

test "compiler AST lowers match constructor-style arm patterns as binding patterns" {
    const source_text =
        \\error Failure;
        \\pub fn run(value: Result<u256, Failure>) -> u256 {
        \\    match (value) {
        \\        Ok(inner) => {
        \\            return inner;
        \\        }
        \\        Err(err) => {
        \\            return 0;
        \\        }
        \\        else => {
        \\            return 1;
        \\        }
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const match_stmt = ast_file.statement(body.statements[0]).Switch;

    try testing.expect(match_stmt.arms.len >= 2);
    try testing.expect(match_stmt.else_body != null);

    switch (match_stmt.arms[0].pattern) {
        .Ok => |pattern_id| try testing.expect(ast_file.pattern(pattern_id).* == .Name),
        else => return error.TestUnexpectedResult,
    }
    switch (match_stmt.arms[1].pattern) {
        .Err => |pattern_id| try testing.expect(ast_file.pattern(pattern_id).* == .Name),
        else => return error.TestUnexpectedResult,
    }
}

test "compiler lowers error-union match statements with ok and err bindings" {
    const source_text =
        \\error Failure;
        \\pub fn run(value: Result<u256, Failure>) -> u256 {
        \\    match (value) {
        \\        Ok(inner) => {
        \\            return inner;
        \\        }
        \\        Err(err) => {
        \\            return 0;
        \\        }
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.is_error") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.unwrap") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.switch") == null);
}

test "compiler lowers error-union match expressions with ok and err bindings" {
    const source_text =
        \\error Failure(code: u256);
        \\pub fn run(value: Result<u256, Failure>) -> u256 {
        \\    let out = match (value) {
        \\        Ok(inner) => inner,
        \\        Err(err) => err.code,
        \\    };
        \\    return out;
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.is_error") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.unwrap") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.get_error") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.switch_expr") == null);
}

test "compiler rejects non-exhaustive error-union match expressions without else" {
    const source_text =
        \\error Failure;
        \\pub fn run(value: Result<u256, Failure>) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "match on Result/error union must cover both Ok(...) and Err(...), or provide else"));
}

test "compiler rejects non-exhaustive tag-only enum match expressions without else" {
    const source_text =
        \\enum State {
        \\    Open,
        \\    Closed,
        \\}
        \\
        \\pub fn run(state: State) -> u256 {
        \\    return match (state) {
        \\        State.Open => 1,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "match on enum must cover all variants or provide else"));
}

test "compiler rejects non-exhaustive payload enum match expressions without else" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\pub fn run(event: Event) -> u256 {
        \\    return match (event) {
        \\        Event.Value(value) => value,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "match on enum must cover all variants or provide else"));
}

test "compiler warns when enum wildcard covers named variants" {
    const source_text =
        \\enum State {
        \\    Open,
        \\    Closed,
        \\}
        \\
        \\pub fn run(state: State) -> u256 {
        \\    return match (state) {
        \\        State.Open => 1,
        \\        else => 2,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), diagnosticSeverityCount(&typecheck.diagnostics, .Error));
    try testing.expect(diagnosticContains(&typecheck.diagnostics, .Warning, "wildcard match arm covers named variants"));
}

test "compiler does not warn when enum variants are explicit before else" {
    const source_text =
        \\enum State {
        \\    Open,
        \\    Closed,
        \\}
        \\
        \\pub fn run(state: State) -> u256 {
        \\    return match (state) {
        \\        State.Open => 1,
        \\        State.Closed => 2,
        \\        else => 3,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), diagnosticSeverityCount(&typecheck.diagnostics, .Error));
    try testing.expect(!diagnosticContains(&typecheck.diagnostics, .Warning, "wildcard match arm covers named variants"));
}

test "compiler warns when Result wildcard covers named error variants" {
    const source_text =
        \\error Failure;
        \\pub fn run(value: Result<u256, Failure>) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        else => 0,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), diagnosticSeverityCount(&typecheck.diagnostics, .Error));
    try testing.expect(diagnosticContains(&typecheck.diagnostics, .Warning, "wildcard match arm covers named variants"));
}

test "compiler rejects mixed Result match patterns and ordinary value patterns" {
    const source_text =
        \\error Failure;
        \\pub fn run(value: Result<u256, Failure>) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        0 => 0,
        \\        Err(err) => 1,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "cannot mix Ok(...)/Err(...)/named error match arms with ordinary value/range patterns"));
}

test "compiler allows Err(binding) on multi-error Result as opaque binding" {
    const source_text =
        \\error Failure(code: u256);
        \\error Denied(owner: address);
        \\fn run(flag: bool, value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Err(err) => 0,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler rejects field access on opaque multi-error Err binding" {
    const source_text =
        \\error Failure(code: u256);
        \\error Denied(owner: address);
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Err(err) => err.code,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "Err(binding) over multiple possible error types is opaque; field access is not supported"));
}

test "compiler allows payloadless named error arms on multi-error Result" {
    const source_text =
        \\error Failure;
        \\error Denied;
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Failure => 7,
        \\        Denied => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.get_error") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "arith.cmpi eq") != null);
}

test "compiler treats named error or-patterns as exhaustive on multi-error Result" {
    const source_text =
        \\error Failure;
        \\error Denied;
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Failure | Denied => 0,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.match_expr") != null or
        std.mem.indexOf(u8, rendered, "scf.") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "placeholder") == null);
}

test "compiler allows payload-carrying named error arms on multi-error Result" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\error Denied(owner: address);
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Failure(code, owner) => code,
        \\        Denied(owner) => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.get_error") != null);
}

test "compiler lowers multi-field named error match arms through OraToSIR" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\error Denied(owner: address);
        \\
        \\contract Probe {
        \\    fn decide(flag: u256) -> !u256 | Failure | Denied {
        \\        if (flag == 0) {
        \\            return 10;
        \\        }
        \\        if (flag == 1) {
        \\            return Failure(7, 0x0000000000000000000000000000000000000000);
        \\        }
        \\        return Denied(0x0000000000000000000000000000000000000000);
        \\    }
        \\
        \\    pub fn run(flag: u256) -> u256 {
        \\        return match (decide(flag)) {
        \\            Ok(inner) => inner,
        \\            Failure(code, owner) => code,
        \\            Denied(owner) => 9,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decide:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn run:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
}

test "compiler supports discard patterns in Result match arms" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\pub fn run(value: !u256 | Failure) -> u256 {
        \\    return match (value) {
        \\        Ok(_) => 1,
        \\        Failure(_, owner) => 2,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.is_error") != null);
}

test "compiler supports Err discard patterns on multi-error Result" {
    const source_text =
        \\error Failure(code: u256);
        \\error Denied(owner: address);
        \\pub fn run(value: !u256 | Failure | Denied) -> u256 {
        \\    return match (value) {
        \\        Ok(_) => 1,
        \\        Err(_) => 2,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.is_error") != null);
}

test "compiler lowers discard patterns in named error match arms through OraToSIR" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\
        \\contract Probe {
        \\    fn decide(flag: u256) -> !u256 | Failure {
        \\        if (flag == 0) {
        \\            return 10;
        \\        }
        \\        return Failure(7, 0x0000000000000000000000000000000000000000);
        \\    }
        \\
        \\    pub fn run(flag: u256) -> u256 {
        \\        return match (decide(flag)) {
        \\            Ok(_) => 1,
        \\            Failure(_, owner) => 2,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn run:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "=>"));
}

test "compiler rejects try expressions in non-Result-returning functions" {
    const source_text =
        \\error Failure;
        \\fn helper(flag: bool) -> Result<u256, Failure> {
        \\    if (flag) return Ok(1);
        \\    return Err(Failure());
        \\}
        \\fn consume(flag: bool) -> u256 {
        \\    return try helper(flag);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "try expression requires a function that returns Result/error union"));
}

test "compiler allows err binding match on multi-error unions as opaque binding" {
    const source_text =
        \\error A;
        \\error B;
        \\pub fn run(value: !u256 | A | B) -> u256 {
        \\    return match (value) {
        \\        Ok(inner) => inner,
        \\        Err(err) => 0,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler resolves Result<T, E> as an error-union-compatible type" {
    const source_text =
        \\error Failure;
        \\pub fn lift(flag: bool, value: u256) -> Result<u256, Failure> {
        \\    if (flag) {
        \\        return value;
        \\    }
        \\    return error.Failure();
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.ok") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.return") != null);
}

test "compiler lowers Result constructors in declaration and return position" {
    const source_text =
        \\error Failure(code: u256);
        \\pub fn make_ok() -> Result<u256, Failure> {
        \\    let value: Result<u256, Failure> = Ok(7);
        \\    return value;
        \\}
        \\
        \\pub fn make_err(flag: bool) -> Result<u256, Failure> {
        \\    if (flag) {
        \\        return Ok(1);
        \\    }
        \\    let value: Result<u256, Failure> = Err(Failure(9));
        \\    return value;
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.ok") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.err") != null or
        std.mem.indexOf(u8, rendered, "ora.error.return") != null);
}

test "compiler lowers Result constructors in match expression arms" {
    const source_text =
        \\error Failure(code: u256);
        \\contract Sample {
        \\    fn choose(flag: bool, value: u256) -> Result<u256, Failure> {
        \\        if (flag) {
        \\            return Ok(value);
        \\        }
        \\        return Err(Failure(7));
        \\    }
        \\
        \\    pub fn pass_through(flag: bool, value: u256) -> Result<u256, Failure> {
        \\        let maybe = choose(flag, value);
        \\        return match (maybe) {
        \\            Ok(inner) => Ok(inner),
        \\            Failure(code) => Err(Failure(code)),
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "func.call @Ok") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "func.call @Err") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.ok") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.return") != null);
}

test "compiler emits error-union ABI attrs for public Result returns" {
    const source_text =
        \\error Failure();
        \\
        \\contract Vault {
        \\    pub fn quote(flag: bool, amount: u256) -> Result<u256, Failure> {
        \\        if (flag) {
        \\            return Ok(amount);
        \\        }
        \\        return Err(Failure());
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"uint256\""));
}

test "compiler emits tagged ABI attrs for supported public Result parameters" {
    const source_text =
        \\error Failure();
        \\
        \\contract Vault {
        \\    pub fn run(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_params"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,uint256)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.result_input_modes"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_payloadless"));
}

test "compiler emits wide tagged ABI attrs for payload-carrying public Result parameters" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Vault {
        \\    pub fn run(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,uint256,uint256)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_single_error"));
}

test "compiler emits wide payloadless ABI attrs for multi-word public Result parameters" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\error Failure();
        \\
        \\contract Vault {
        \\    pub fn run(value: Result<Pair, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner.left,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,(uint256,uint256))\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_payloadless"));
}

test "compiler lowers std.result helpers for Result values" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\error Failure();
        \\
        \\fn choose(flag: bool, value: u256) -> Result<u256, Failure> {
        \\    if (flag) {
        \\        return Ok(value);
        \\    }
        \\    return Err(Failure());
        \\}
        \\
        \\pub fn run(flag: bool, value: u256) -> u256 {
        \\    let maybe = choose(flag, value);
        \\    let ok = std.result.is_ok(maybe);
        \\    let err = std.result.is_err(maybe);
        \\    let base = std.result.unwrap_or(maybe, 7);
        \\    if (ok and !err) {
        \\        return base;
        \\    }
        \\    return 9;
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "call @std.result.is_ok__u256__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "call @std.result.is_err__u256__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "call @std.result.unwrap_or__u256__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "func.func @std.result.is_ok__u256__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "func.func @std.result.is_err__u256__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "func.func @std.result.unwrap_or__u256__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.is_error") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.unwrap") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.if") != null);
}

test "compiler supports std.result.unwrap_or on dynamic Result payloads" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\error Failure();
        \\
        \\fn choose(flag: bool, value: bytes) -> Result<bytes, Failure> {
        \\    if (flag) {
        \\        return Ok(value);
        \\    }
        \\    return Err(Failure());
        \\}
        \\
        \\pub fn run(flag: bool, value: bytes) -> bytes {
        \\    let maybe = choose(flag, value);
        \\    return std.result.unwrap_or(maybe, value);
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "call @std.result.unwrap_or__bytes__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "func.func @std.result.unwrap_or__bytes__Failure") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.unwrap") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.if") != null);
}

test "compiler converts imported std.result helpers through OraToSIR" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\error Failure();
        \\
        \\fn choose(flag: bool, value: u256) -> Result<u256, Failure> {
        \\    if (flag) {
        \\        return Ok(value);
        \\    }
        \\    return Err(Failure());
        \\}
        \\
        \\pub fn run(flag: bool, value: u256) -> u256 {
        \\    let maybe = choose(flag, value);
        \\    let ok = std.result.is_ok(maybe);
        \\    let err = std.result.is_err(maybe);
        \\    let base = std.result.unwrap_or(maybe, 7);
        \\    if (ok and !err) {
        \\        return base;
        \\    }
        \\    return 9;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "icall @choose"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn std_result_is_ok__u256__Failure:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn std_result_is_err__u256__Failure:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn std_result_unwrap_or__u256__Failure:"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler converts imported std.result dynamic payload helpers through OraToSIR" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\error Failure();
        \\
        \\fn choose_bytes(flag: bool, value: bytes) -> Result<bytes, Failure> {
        \\    if (flag) {
        \\        return Ok(value);
        \\    }
        \\    return Err(Failure());
        \\}
        \\
        \\pub fn run(flag: bool, value: bytes) -> bytes {
        \\    let maybe = choose_bytes(flag, value);
        \\    return std.result.unwrap_or(maybe, value);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "icall @choose_bytes"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn std_result_unwrap_or__bytes__Failure:"));
    try expectNoResidualOraRuntimeOps(rendered);
}

test "compiler rejects public Result parameters when one-word success would require multi-word error carrier" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\
        \\contract Vault {
        \\    pub fn run(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => err.owner,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expect(diags.items.items.len != 0);
    var found = false;
    for (diags.items.items) |diag| {
        if (std.mem.indexOf(u8, diag.message, "unsupported Result ABI type") != null and
            std.mem.indexOf(u8, diag.message, "carrier-compatible payload and a single error") != null)
        {
            found = true;
            break;
        }
    }
    try testing.expect(found);
}

test "compiler keeps plain match expressions on booleans on the switch path" {
    const source_text =
        \\pub fn run(flag: bool) -> u256 {
        \\    let value = match (flag) {
        \\        true => 1,
        \\        else => 0,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);
    try testing.expect(
        std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr") or
            std.mem.containsAtLeast(u8, hir_text, 1, "ora.match_expr") or
            std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"),
    );
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

test "compiler allows opaque multi-error catch bindings when unused" {
    const source_text =
        \\error ErrorA(code: u256);
        \\error ErrorB(required: u256);
        \\
        \\pub fn handle(maybe: !u256 | ErrorA | ErrorB) -> u256 {
        \\    try {
        \\        maybe;
        \\    } catch (e) {
        \\        return 0;
        \\    }
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
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

test "compiler source loader injects embedded std result module" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\error Failure;
        \\
        \\fn choose(flag: bool, value: u256) -> Result<u256, Failure> {
        \\    if (flag) {
        \\        return Ok(value);
        \\    }
        \\    return Err(Failure);
        \\}
        \\
        \\pub fn run(flag: bool, value: u256) -> bool {
        \\    return std.result.is_err(choose(flag, value));
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const package = compilation.db.sources.package(compilation.package_id);
    try testing.expectEqual(@as(usize, 5), package.modules.items.len);

    const graph = try compilation.db.moduleGraph(compilation.package_id);
    const root_summary = for (graph.modules) |summary| {
        if (summary.module_id == compilation.root_module_id) break summary;
    } else return error.TestUnexpectedResult;

    try testing.expectEqual(@as(usize, 1), root_summary.imports.len);
    const std_module_id = root_summary.imports[0].target_module_id orelse return error.TestUnexpectedResult;
    const std_summary = for (graph.modules) |summary| {
        if (summary.module_id == std_module_id) break summary;
    } else return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 3), std_summary.imports.len);
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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.error_union<!ora.tuple<i256, i1>"));
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
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "declaration expects type '[u256; 3]', found '[integer; 4]'"));
}

test "compiler abi emits enum wire type matching declared repr width" {
    const source_text =
        \\enum Status: u8 { Active, Paused }
        \\
        \\contract C {
        \\    pub fn set(status: Status) -> Status {
        \\        return status;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_set = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "set")) continue;
        try testing.expectEqual(@as(usize, 1), callable.inputs.len);
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        const input = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("uint8", input.wire_type.?);
        try testing.expectEqualStrings("uint8", output.wire_type.?);
        const repr = contract_abi.findType(input.repr_type.?) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("uint8", repr.wire_type.?);
        saw_set = true;
    }
    try testing.expect(saw_set);
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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct_anon<"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.muli"));
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

test "compiler uses selector-derived ids for imported Result errors" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
        \\error Failure;
        \\
        \\pub fn choose(flag: bool) -> Result<u256, Failure> {
        \\    if (flag) {
        \\        return Err(Failure);
        \\    }
        \\    return Ok(1);
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\pub contract Main {
        \\    pub fn run(flag: bool) -> u256 {
        \\        return match (dep.choose(flag)) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
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
    const rendered = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(rendered);

    const selector_id = compiler.hir.abi.keccakSelectorValue("Failure()");
    const selector_text = try std.fmt.allocPrint(testing.allocator, "ora.error_id = {d}", .{selector_id});
    defer testing.allocator.free(selector_text);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sym_name = \"dep.Failure\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, selector_text));
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

test "compiler emits dispatcher metadata for public Result inputs" {
    const source_text =
        \\error Failure();
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,uint256)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.result_input_modes"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_payloadless"));
}

test "compiler emits dispatcher metadata for payload-carrying public Result inputs" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,uint256,uint256)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_single_error"));
}

test "compiler emits dispatcher metadata for multi-word payloadless public Result inputs" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\error Failure();
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<Pair, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner.left,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,(uint256,uint256))\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_payloadless"));
}

test "compiler emits dispatcher metadata for multi-word public Result inputs" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\error Failure(code: u256, owner: address);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<Pair, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner.left,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,(uint256,uint256),(uint256,address))\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_single_error"));
}

test "compiler emits dispatcher metadata for dynamic bytes public Result inputs" {
    const source_text =
        \\error Failure();
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<bytes, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => 1,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,bytes)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_payloadless"));
}

test "compiler emits dispatcher metadata for dynamic bytes public Result inputs with payload errors" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<bytes, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => 1,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,bytes,uint256)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_single_error"));
}

test "compiler lowers payloadless public Result inputs through OraToSIR dispatcher path" {
    const source_text =
        \\error Failure();
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn consume:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "0x9E4DA34B"));
}

test "compiler lowers payload-carrying public Result inputs through OraToSIR dispatcher path" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn consume:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
}

test "compiler lowers multi-word payloadless public Result inputs through OraToSIR dispatcher path" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\error Failure();
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<Pair, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner.left,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn consume:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
}

test "compiler lowers multi-word public Result inputs with static error payloads through OraToSIR dispatcher path" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\error Failure(code: u256, owner: address);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<Pair, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner.left,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn consume:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
}

test "compiler lowers dynamic bytes public Result inputs through OraToSIR dispatcher path" {
    const source_text =
        \\error Failure();
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<bytes, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => 1,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn consume:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "calldatacopy"));
}

test "compiler lowers dynamic bytes public Result inputs with payload errors through OraToSIR dispatcher path" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<bytes, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => 1,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn consume:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "calldatacopy"));
}

test "compiler emits dispatcher metadata for dynamic slice public Result inputs with payload errors" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<slice[u256], Failure>) -> u256 {
        \\        let total = 0;
        \\        match (value) {
        \\            Ok(inner) => {
        \\                for (inner) |item| {
        \\                    total = total + item;
        \\                }
        \\            },
        \\            Err(err) => {
        \\                total = err.code;
        \\            }
        \\        }
        \\        return total;
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(bool,uint256[],uint256)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "wide_single_error"));
}

test "compiler lowers dynamic slice public Result inputs with payload errors through OraToSIR dispatcher path" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn consume(value: Result<slice[u256], Failure>) -> u256 {
        \\        let total = 0;
        \\        match (value) {
        \\            Ok(inner) => {
        \\                for (inner) |item| {
        \\                    total = total + item;
        \\                }
        \\            },
        \\            Err(err) => {
        \\                total = err.code;
        \\            }
        \\        }
        \\        return total;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn consume:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn main:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "calldatacopy"));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.ok"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.err"));
}

test "compiler keeps wide error payloads out of narrow packed error-union carrier" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn run(flag: bool, code: u256) -> !bool | Failure {
        \\        if (flag) {
        \\            return Failure(code);
        \\        }
        \\        return false;
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.addptr"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.error.return"));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.yield"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.conditional_return"));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(address,string)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.addptr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.load"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "public error-union dispatcher currently supports scalar, static tuple/struct, bytes/string, and static-base dynamic array ABI success payloads"));
}

test "complex SMT app probes match between sequential and parallel verification" {
    const probes = [_]struct { path: []const u8, function_name: []const u8, timeout_ms: u32 }{
        .{ .path = "ora-example/apps/defi_lending_pool.ora", .function_name = "calculate_utilization_rate", .timeout_ms = 5_000 },
        .{ .path = "ora-example/apps/defi_lending_pool.ora", .function_name = "get_available_liquidity", .timeout_ms = 5_000 },
        .{ .path = "ora-example/apps/erc20_bitfield_comptime_generics.ora", .function_name = "transfer", .timeout_ms = 5_000 },
        .{ .path = "ora-example/smt/soundness/conditional_return_split.ora", .function_name = "withdraw", .timeout_ms = 5_000 },
        .{ .path = "ora-example/smt/soundness/overflow_mul_constant.ora", .function_name = "percentOf", .timeout_ms = 15_000 },
        .{ .path = "ora-example/smt/soundness/switch_arm_path_predicates.ora", .function_name = "categorize", .timeout_ms = 5_000 },
        .{ .path = "ora-example/smt/soundness/fail_loop_invariant_post.ora", .function_name = "countTo", .timeout_ms = 5_000 },
    };

    for (probes) |probe| {
        var seq_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, false, probe.timeout_ms);
        defer seq_result.deinit(testing.allocator);

        var par_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, true, probe.timeout_ms);
        defer par_result.deinit(testing.allocator);

        try testing.expect(!seq_result.degraded);
        try testing.expect(!par_result.degraded);
        try expectVerificationProbeEquivalent(&seq_result, &par_result);
    }
}

test "SMT release matrix probes match between sequential and parallel verification" {
    const probes = [_]struct { path: []const u8, function_name: []const u8 }{
        .{ .path = "ora-example/smt/guards/exact_proven.ora", .function_name = "provenExactDivisionReturn" },
        .{ .path = "ora-example/smt/verification/function_contracts_basic.ora", .function_name = "deposit" },
        .{ .path = "ora-example/smt/verification/function_contracts_old.ora", .function_name = "incrementAndReturn" },
        .{ .path = "ora-example/smt/verification/ghost_variables.ora", .function_name = "deposit" },
        .{ .path = "ora-example/smt/verification/ghost_combined.ora", .function_name = "deposit" },
        .{ .path = "ora-example/smt/verification/loop_invariants.ora", .function_name = "accumulateToStorage" },
        .{ .path = "ora-example/smt/summaries/callee_state_effects.ora", .function_name = "doIncrement" },
        .{ .path = "ora-example/comptime/generics/generic_struct_multi_type_params.ora", .function_name = "value_plus_one" },
        .{ .path = "ora-example/comptime/generics/generic_fn_control_flow_clone.ora", .function_name = "rem_u256" },
    };

    for (probes) |probe| {
        var seq_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, false, 5_000);
        defer seq_result.deinit(testing.allocator);

        var par_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, true, 5_000);
        defer par_result.deinit(testing.allocator);

        try testing.expect(!seq_result.degraded);
        try testing.expect(!par_result.degraded);
        try testing.expect(seq_result.success);
        try testing.expect(par_result.success);
        try testing.expectEqual(@as(usize, 0), seq_result.errors_len);
        try testing.expectEqual(@as(usize, 0), par_result.errors_len);
        try expectVerificationProbeEquivalent(&seq_result, &par_result);
    }
}

test "refined aggregate probes match between sequential and parallel verification" {
    const probes = [_]struct { path: []const u8, function_name: []const u8 }{
        .{ .path = "ora-example/corpus/types/refinement/refinement_struct_field_proof.ora", .function_name = "build" },
        .{ .path = "ora-example/corpus/types/refinement/refinement_adt_payload_proof.ora", .function_name = "build" },
    };

    for (probes) |probe| {
        var seq_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, false, 5_000);
        defer seq_result.deinit(testing.allocator);

        var par_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, true, 5_000);
        defer par_result.deinit(testing.allocator);

        try testing.expect(seq_result.success);
        try testing.expect(par_result.success);
        try testing.expect(!seq_result.degraded);
        try testing.expect(!par_result.degraded);
        try testing.expectEqual(@as(usize, 0), seq_result.errors_len);
        try testing.expectEqual(@as(usize, 0), par_result.errors_len);
        try testing.expectEqual(@as(usize, 1), seq_result.diagnostics_len);
        try testing.expectEqual(@as(usize, 1), par_result.diagnostics_len);
        try expectVerificationProbeEquivalent(&seq_result, &par_result);
    }
}

test "SMT expected-failure probes match between sequential and parallel verification" {
    const probes = [_]struct {
        path: []const u8,
        function_name: []const u8,
        expected_error_kinds: []const u8,
    }{
        .{
            .path = "ora-example/smt/verification/ora_assert_obligation_fail.ora",
            .function_name = "alwaysFails",
            .expected_error_kinds = "InvariantViolation",
        },
    };

    for (probes) |probe| {
        var seq_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, false, 5_000);
        defer seq_result.deinit(testing.allocator);

        var par_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, true, 5_000);
        defer par_result.deinit(testing.allocator);

        try testing.expect(!seq_result.degraded);
        try testing.expect(!par_result.degraded);
        try testing.expect(seq_result.errors_len > 0);
        try testing.expect(par_result.errors_len > 0);
        try testing.expectEqualStrings(probe.expected_error_kinds, seq_result.error_kinds);
        try testing.expectEqualStrings(probe.expected_error_kinds, par_result.error_kinds);
        try expectVerificationProbeEquivalent(&seq_result, &par_result);
    }
}
