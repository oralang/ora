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
