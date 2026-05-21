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

test "compiler abi emits uint256 for enum without declared repr" {
    const source_text =
        \\enum Status { Active, Paused }
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

    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "set")) continue;
        const input = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("uint256", input.wire_type.?);
        try testing.expectEqualStrings("uint256", output.wire_type.?);
        return;
    }
    return error.TestUnexpectedResult;
}

test "compiler abi emits fixed bytes public ABI types" {
    const source_text =
        \\contract C {
        \\    pub fn set(a: bytes1, b: bytes4, c: bytes20, d: bytes31, e: bytes32, value: u256) -> bytes31 {
        \\        return d;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "set")) continue;
        try testing.expectEqual(@as(usize, 6), callable.inputs.len);
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        try testing.expectEqualStrings("0x36a0470c", callable.selector.?);
        const b1_input = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        const b4_input = contract_abi.findType(callable.inputs[1].type_id) orelse return error.TestUnexpectedResult;
        const b20_input = contract_abi.findType(callable.inputs[2].type_id) orelse return error.TestUnexpectedResult;
        const b31_input = contract_abi.findType(callable.inputs[3].type_id) orelse return error.TestUnexpectedResult;
        const b32_input = contract_abi.findType(callable.inputs[4].type_id) orelse return error.TestUnexpectedResult;
        const value_input = contract_abi.findType(callable.inputs[5].type_id) orelse return error.TestUnexpectedResult;
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("bytes1", b1_input.wire_type.?);
        try testing.expectEqualStrings("bytes4", b4_input.wire_type.?);
        try testing.expectEqualStrings("bytes20", b20_input.wire_type.?);
        try testing.expectEqualStrings("bytes31", b31_input.wire_type.?);
        try testing.expectEqualStrings("bytes32", b32_input.wire_type.?);
        try testing.expectEqualStrings("uint256", value_input.wire_type.?);
        try testing.expectEqualStrings("bytes31", output.wire_type.?);
        return;
    }
    return error.TestUnexpectedResult;
}

test "compiler abi emits registry-backed refinement predicates" {
    const source_text =
        \\contract C {
        \\    pub fn configure(rate: BasisPoints<u256>, amount: NonZero<u256>, owner: NonZeroAddress) -> BasisPoints<u256> {
        \\        return rate;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "configure")) continue;
        try testing.expectEqual(@as(usize, 3), callable.inputs.len);
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);

        const rate = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        const amount = contract_abi.findType(callable.inputs[1].type_id) orelse return error.TestUnexpectedResult;
        const owner = contract_abi.findType(callable.inputs[2].type_id) orelse return error.TestUnexpectedResult;
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;

        try testing.expectEqualStrings("{\"kind\":\"range\",\"min\":\"0\",\"max\":\"10000\"}", rate.predicate_json.?);
        try testing.expectEqualStrings("{\"kind\":\"nonZero\"}", amount.predicate_json.?);
        try testing.expectEqualStrings("{\"kind\":\"nonZeroAddress\"}", owner.predicate_json.?);
        try testing.expectEqualStrings("{\"kind\":\"range\",\"min\":\"0\",\"max\":\"10000\"}", output.predicate_json.?);
        try testing.expectEqualStrings("uint256", rate.wire_type.?);
        try testing.expectEqualStrings("uint256", amount.wire_type.?);
        try testing.expectEqualStrings("address", owner.wire_type.?);
        return;
    }
    return error.TestUnexpectedResult;
}

test "compiler abi keeps anonymous structs distinct from tuples" {
    const source_text =
        \\contract Test {
        \\    pub fn viewFn() -> struct { amount: u256, ok: bool } {
        \\        return .{ .amount = 1, .ok = true };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_struct_output = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "viewFn")) continue;
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(usize, 2), output.fields.len);
        try testing.expectEqual(@as(usize, 0), output.components.len);
        try testing.expectEqualStrings("amount", output.fields[0].name);
        try testing.expectEqualStrings("ok", output.fields[1].name);
        saw_struct_output = true;
    }
    try testing.expect(saw_struct_output);
}

test "compiler abi preserves named struct return types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    pub fn viewFn() -> Pair {
        \\        return Pair { left: 1, right: true };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_struct_output = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "viewFn")) continue;
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", output.name.?);
        try testing.expectEqual(@as(usize, 2), output.fields.len);
        try testing.expectEqual(@as(usize, 0), output.components.len);
        try testing.expectEqualStrings("left", output.fields[0].name);
        try testing.expectEqualStrings("right", output.fields[1].name);
        saw_struct_output = true;
    }
    try testing.expect(saw_struct_output);
}

test "compiler abi preserves named struct parameter types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    pub fn consume(pair: Pair) -> bool {
        \\        return pair.right;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_struct_input = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "consume")) continue;
        try testing.expectEqual(@as(usize, 1), callable.inputs.len);
        const input = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", input.name.?);
        try testing.expectEqual(@as(usize, 2), input.fields.len);
        try testing.expectEqual(@as(usize, 0), input.components.len);
        try testing.expectEqualStrings("left", input.fields[0].name);
        try testing.expectEqualStrings("right", input.fields[1].name);
        saw_struct_input = true;
    }
    try testing.expect(saw_struct_input);
}

test "compiler abi preserves nested named struct return types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    struct Wrapper {
        \\        pair: Pair,
        \\        ok: bool,
        \\    }
        \\
        \\    pub fn viewFn() -> Wrapper {
        \\        return Wrapper { pair: Pair { left: 1, right: true }, ok: true };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_wrapper_output = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "viewFn")) continue;
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        const wrapper = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Wrapper", wrapper.name.?);
        try testing.expectEqual(@as(usize, 2), wrapper.fields.len);
        try testing.expectEqualStrings("pair", wrapper.fields[0].name);
        try testing.expectEqualStrings("ok", wrapper.fields[1].name);
        try testing.expectEqual(@as(usize, 0), wrapper.components.len);

        const pair = contract_abi.findType(wrapper.fields[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", pair.name.?);
        try testing.expectEqual(@as(usize, 2), pair.fields.len);
        try testing.expectEqualStrings("left", pair.fields[0].name);
        try testing.expectEqualStrings("right", pair.fields[1].name);
        try testing.expectEqual(@as(usize, 0), pair.components.len);
        saw_wrapper_output = true;
    }
    try testing.expect(saw_wrapper_output);
}

test "compiler abi preserves nested named struct parameter types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    struct Wrapper {
        \\        pair: Pair,
        \\        ok: bool,
        \\    }
        \\
        \\    pub fn consume(wrapper: Wrapper) -> bool {
        \\        return wrapper.pair.right;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_wrapper_input = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "consume")) continue;
        try testing.expectEqual(@as(usize, 1), callable.inputs.len);
        const wrapper = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Wrapper", wrapper.name.?);
        try testing.expectEqual(@as(usize, 2), wrapper.fields.len);
        try testing.expectEqualStrings("pair", wrapper.fields[0].name);
        try testing.expectEqualStrings("ok", wrapper.fields[1].name);
        try testing.expectEqual(@as(usize, 0), wrapper.components.len);

        const pair = contract_abi.findType(wrapper.fields[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", pair.name.?);
        try testing.expectEqual(@as(usize, 2), pair.fields.len);
        try testing.expectEqualStrings("left", pair.fields[0].name);
        try testing.expectEqualStrings("right", pair.fields[1].name);
        try testing.expectEqual(@as(usize, 0), pair.components.len);
        saw_wrapper_input = true;
    }
    try testing.expect(saw_wrapper_input);
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

test "compiler const eval supports selector builtin for trait methods" {
    const source_text =
        \\trait ERC20 {
        \\    fn transfer(self, to: address, value: u256) -> bool;
        \\}
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        @selector(ERC20.transfer);
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
    const selector_value = consteval.values[ret_stmt.value.?.index()].?.fixed_bytes;
    try testing.expectEqualSlices(u8, &.{ 0xa9, 0x05, 0x9c, 0xbb }, selector_value);
}

test "compiler const eval supports abiSignature builtin for trait methods" {
    const source_text =
        \\trait ERC20 {
        \\    fn transfer(self, to: address, value: u256) -> bool;
        \\}
        \\pub fn run() -> string {
        \\    return comptime {
        \\        @abiSignature(ERC20.transfer);
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
    try testing.expectEqualStrings("transfer(address,uint256)", consteval.values[ret_stmt.value.?.index()].?.string);
}

test "compiler selector builtin rejects non-function arguments" {
    const source_text =
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        @selector(42);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "function reference"));
}

test "compiler selector builtin integrates with bytes4 assignments and comparison" {
    const source_text =
        \\trait ERC20 {
        \\    fn transfer(self, to: address, value: u256) -> bool;
        \\}
        \\pub fn run() -> bool {
        \\    return comptime {
        \\        const sel: bytes4 = @selector(ERC20.transfer);
        \\        sel == @selector(ERC20.transfer);
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
    const comptime_body = ast_file.body(ast_file.expression(ret_stmt.value.?).Comptime.body);
    const decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.pattern_types[decl.pattern.index()].kind());
    try testing.expectEqual(@as(u8, 4), typecheck.pattern_types[decl.pattern.index()].type.fixed_bytes.len);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0xa9, 0x05, 0x9c, 0xbb }, consteval.values[decl.value.?.index()].?.fixed_bytes);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval supports bytewise fixed-bytes xor" {
    const source_text =
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        const lhs: bytes4 = hex"01020304";
        \\        const rhs: bytes4 = hex"05060708";
        \\        lhs ^ rhs;
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

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 4), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0x04, 0x04, 0x04, 0x0c }, consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler trait method reflection exposes selectors" {
    const source_text =
        \\trait ERC165 {
        \\    fn supportsInterface(self, interface_id: bytes4) -> bool;
        \\}
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        const selector: bytes4 = @traitMethods(ERC165)[0].selector;
        \\        selector;
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
    const comptime_body = ast_file.body(ast_file.expression(ret_stmt.value.?).Comptime.body);
    const selector_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 4), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0x01, 0xff, 0xc9, 0xa7 }, consteval.values[selector_decl.value.?.index()].?.fixed_bytes);
}

test "compiler const eval supports eventTopic builtin for log declarations" {
    const source_text =
        \\log Transfer(indexed from: address, indexed to: address, value: u256);
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(Transfer);
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

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 32), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eventTopic builtin rejects non-event arguments" {
    const source_text =
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(42);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "event reference"));
}

test "compiler eventTopic builtin supports contract-qualified log references" {
    const source_text =
        \\contract Events {
        \\    log Transfer(indexed from: address, indexed to: address, value: u256);
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(Events.Transfer);
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

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eventTopic builtin uses pinned event_name metadata" {
    const source_text =
        \\log TransferV2(indexed from: address, indexed to: address, value: u256) {
        \\    pub const event_name = "Transfer";
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(TransferV2);
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

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eventTopic builtin rejects non-string event_name metadata" {
    const source_text =
        \\log Transfer(indexed from: address, indexed to: address, value: u256) {
        \\    pub const event_name = 42;
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(Transfer);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "event_name"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "string literal"));
}

test "compiler preserves pinned eip712_name metadata on structs" {
    const source_text =
        \\struct PermitV2 {
        \\    pub const eip712_name = "Permit";
        \\    owner: address,
        \\    spender: address,
        \\    value: u256,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(@as(usize, 1), struct_item.metadata.len);
    try testing.expectEqualStrings("Permit", compiler.hir.abi.eip712WireNameFromStructItem(ast_file, struct_item).?);
}

test "compiler eip712_name metadata falls back to struct identifier" {
    const source_text =
        \\struct Permit {
        \\    owner: address,
        \\    spender: address,
        \\    value: u256,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;

    try testing.expectEqualStrings("Permit", compiler.hir.abi.eip712WireNameFromStructItem(ast_file, struct_item).?);
}

test "compiler rejects non-string eip712_name metadata" {
    const source_text =
        \\struct Permit {
        \\    pub const eip712_name = 42;
        \\    owner: address,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "eip712_name"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "string literal"));
}

test "compiler const eval supports eip712TypeHash builtin for ERC-2612 Permit" {
    const source_text =
        \\struct Permit {
        \\    pub const eip712_name = "Permit";
        \\    owner: address,
        \\    spender: address,
        \\    value: u256,
        \\    nonce: u256,
        \\    deadline: u256,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(Permit);
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

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 32), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eip712TypeHash builtin falls back to struct identifier" {
    const source_text =
        \\struct Permit {
        \\    owner: address,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(Permit);
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

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Permit(address owner)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eip712TypeHash builtin uses pinned eip712_name metadata" {
    const source_text =
        \\struct PermitV2 {
        \\    pub const eip712_name = "Permit";
        \\    owner: address,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(PermitV2);
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

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Permit(address owner)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eip712TypeHash builtin rejects non-struct arguments" {
    const source_text =
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(42);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "struct type"));
}

test "compiler eip712TypeHash builtin rejects nested struct fields" {
    const source_text =
        \\struct Permit {
        \\    owner: address,
        \\}
        \\
        \\struct Order {
        \\    permit: Permit,
        \\    deadline: u256,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(Order);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "nested struct fields"));
}

test "compiler corpus covers ABI selector and signature builtins" {
    var compilation = try compilePackage("ora-example/corpus/comptime/abi_builtins.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("AbiBuiltinCorpus").?;
    const contract = ast_file.item(contract_id).Contract;

    var selector_index: ?usize = null;
    var signature_index: ?usize = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;

        const function = item.Function;
        const body = ast_file.body(function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        if (std.mem.eql(u8, function.name, "transfer_selector")) {
            selector_index = ret_stmt.value.?.index();
        } else if (std.mem.eql(u8, function.name, "transfer_signature")) {
            signature_index = ret_stmt.value.?.index();
        }
    }

    try testing.expect(selector_index != null);
    try testing.expect(signature_index != null);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0xa9, 0x05, 0x9c, 0xbb }, consteval.values[selector_index.?].?.fixed_bytes);
    try testing.expectEqualStrings("transfer(address,uint256)", consteval.values[signature_index.?].?.string);
}

test "compiler corpus covers ABI eventTopic builtin" {
    var compilation = try compilePackage("ora-example/corpus/comptime/event_topic.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("EventTopicCorpus").?;
    const contract = ast_file.item(contract_id).Contract;

    var topic_index: ?usize = null;
    var qualified_topic_index: ?usize = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;

        const body = ast_file.body(item.Function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        if (std.mem.eql(u8, item.Function.name, "transfer_topic")) {
            topic_index = ret_stmt.value.?.index();
        } else if (std.mem.eql(u8, item.Function.name, "transfer_topic_qualified")) {
            qualified_topic_index = ret_stmt.value.?.index();
        }
    }

    try testing.expect(topic_index != null);
    try testing.expect(qualified_topic_index != null);

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[topic_index.?].?.fixed_bytes);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[qualified_topic_index.?].?.fixed_bytes);
}

test "compiler corpus covers ABI eip712TypeHash builtin" {
    var compilation = try compilePackage("ora-example/corpus/comptime/eip712_type_hash.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("Eip712TypeHashCorpus").?;
    const contract = ast_file.item(contract_id).Contract;

    var type_hash_index: ?usize = null;
    var qualified_type_hash_index: ?usize = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;

        const body = ast_file.body(item.Function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        if (std.mem.eql(u8, item.Function.name, "permit_type_hash")) {
            type_hash_index = ret_stmt.value.?.index();
        } else if (std.mem.eql(u8, item.Function.name, "permit_type_hash_qualified")) {
            qualified_type_hash_index = ret_stmt.value.?.index();
        }
    }

    try testing.expect(type_hash_index != null);
    try testing.expect(qualified_type_hash_index != null);

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[type_hash_index.?].?.fixed_bytes);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[qualified_type_hash_index.?].?.fixed_bytes);
}

test "compiler corpus covers std interfaceId helper" {
    var compilation = try compilePackage("ora-example/corpus/comptime/interface_id.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("InterfaceIdCorpus").?;
    const contract = ast_file.item(contract_id).Contract;

    var erc165_index: ?usize = null;
    var mini_index: ?usize = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;

        const function = item.Function;
        const body = ast_file.body(function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        if (std.mem.eql(u8, item.Function.name, "erc165_id")) {
            erc165_index = ret_stmt.value.?.index();
        } else if (std.mem.eql(u8, item.Function.name, "mini_id")) {
            mini_index = ret_stmt.value.?.index();
        }
    }

    try testing.expect(erc165_index != null);
    try testing.expect(mini_index != null);

    var balance_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("balanceOf(address)", &balance_hash, .{});
    var transfer_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("transfer(address,uint256)", &transfer_hash, .{});
    const mini_expected = [_]u8{
        balance_hash[0] ^ transfer_hash[0],
        balance_hash[1] ^ transfer_hash[1],
        balance_hash[2] ^ transfer_hash[2],
        balance_hash[3] ^ transfer_hash[3],
    };

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0x01, 0xff, 0xc9, 0xa7 }, consteval.values[erc165_index.?].?.fixed_bytes);
    try testing.expectEqualSlices(u8, mini_expected[0..], consteval.values[mini_index.?].?.fixed_bytes);
}

test "compiler corpus rejects selector misuse" {
    var compilation = try compilePackage("ora-example/corpus/comptime/fail_selector_non_function.ora");
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "function reference"));
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

test "compiler emits ABI attrs for slice of struct containing bitfield" {
    const source_text =
        \\bitfield CustomFlags : u256 {
        \\    enabled: bool @bits(0..1);
        \\    code: u8 @bits(1..6);
        \\    delta: i16 @bits(6..22);
        \\    amount: u32 @bits(22..54);
        \\}
        \\
        \\struct Row {
        \\    head: u256;
        \\    flags: CustomFlags;
        \\    tail: u256;
        \\}
        \\
        \\contract Entries {
        \\    storage var rows: slice[Row];
        \\
        \\    pub fn set(values: slice[Row]) {
        \\        rows = values;
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_params"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(uint256,uint256,uint256)[]\""));
}
