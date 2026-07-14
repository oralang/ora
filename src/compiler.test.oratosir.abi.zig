const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const compileText = common.compileText;
const renderSirTextForModule = common.renderSirTextForModule;
const expectNoResidualOraRuntimeOps = common.expectNoResidualOraRuntimeOps;
const functionSlice = common.functionSlice;
const expectOrderedNeedles = common.expectOrderedNeedles;
const expectStaticAbiDecodeGuardBeforePayloadLoad = common.expectStaticAbiDecodeGuardBeforePayloadLoad;
const expectDynamicAbiDecodeGuardChain = common.expectDynamicAbiDecodeGuardChain;
const expectDynamicAbiDecodeWordGuardChain = common.expectDynamicAbiDecodeWordGuardChain;
const expectMixedDynamicTupleCarrierShape = common.expectMixedDynamicTupleCarrierShape;

test "compiler lowers Err discard patterns through OraToSIR" {
    const source_text =
        \\error Failure(code: u256);
        \\error Denied(owner: address);
        \\
        \\contract Probe {
        \\    fn decide(flag: bool) -> !u256 | Failure | Denied {
        \\        if (flag) {
        \\            return 10;
        \\        }
        \\        return Failure(7);
        \\    }
        \\
        \\    pub fn run(flag: bool) -> u256 {
        \\        return match (decide(flag)) {
        \\            Ok(_) => 1,
        \\            Err(_) => 2,
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
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "c2 = const 0x2"));
}

test "compiler converts runtime abiDecode scalar memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_scalar(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(u256, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_scalar");

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_scalar:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bool memory result with canonical validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bool(payload: bytes) -> bool {
        \\        let decoded = @abiDecode(bool, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => false,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_bool:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "or"));
    const decode_fn = try functionSlice(rendered, "decode_bool");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecodePermissive memory result shapes through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_scalar(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive(u8, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_text(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive(string, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_values(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive(slice[u256], payload);
        \\        return match (decoded) {
        \\            Ok(values) => values[0],
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_text(payload: bytes) -> u256 {
        \\        let decoded = @abiDecodePermissive((u256, string), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    try expectNoResidualOraRuntimeOps(rendered);

    const scalar_fn = try functionSlice(rendered, "decode_scalar");
    const text_fn = try functionSlice(rendered, "decode_text");
    const values_fn = try functionSlice(rendered, "decode_values");
    const pair_text_fn = try functionSlice(rendered, "decode_pair_text");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(scalar_fn);
    try expectDynamicAbiDecodeWordGuardChain(text_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, text_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(values_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_text_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, pair_text_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_text_fn);
}

test "compiler converts runtime abiDecodePermissive mixed dynamic hex literal through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode() -> u256 {
        \\        let decoded = @abiDecodePermissive((u256, string), hex"000000000000000000000000000000000000000000000000000000000000000700000000000000000000000000000000000000000000000000000000000000600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000161ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        \\        return match (decoded) {
        \\            Ok(value) => value.0 + @cast(u256, value.1[0]),
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
}

test "compiler abiDecode N3b4 validates public calldata bool and address params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn accept(flag: bool, owner: address, amount: u8, delta: i8, tag: bytes4) -> bool {
        \\        return true;
        \\    }
        \\    pub fn full(signed: i256, tag: bytes32) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "or"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 5, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "shl"));
    try expectOrderedNeedles(main_fn, &.{ "a_accept = calldataload", "eq a_accept", "eq a_accept", "or" });
    try expectOrderedNeedles(main_fn, &.{ "b_accept = calldataload", "and b_accept", "eq b_accept" });
    try expectOrderedNeedles(main_fn, &.{ "n_accept = calldataload", "and n_accept", "eq n_accept" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept = calldataload", "signextend", "eq arg_accept" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept_0 = calldataload", "shr", "shl", "eq arg_accept_0" });
    try expectOrderedNeedles(main_fn, &.{ "a_accept = calldataload", "or", ": @abi_decode_revert_4", "b_accept = calldataload" });
    try expectOrderedNeedles(main_fn, &.{ "b_accept = calldataload", "eq b_accept", ": @abi_decode_revert_5", "n_accept = calldataload" });
    try expectOrderedNeedles(main_fn, &.{ "n_accept = calldataload", "eq n_accept", ": @abi_decode_revert_3" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept = calldataload", "eq arg_accept", ": @abi_decode_revert_3" });
    try expectOrderedNeedles(main_fn, &.{ "arg_accept_0 = calldataload", "eq arg_accept_0", "? @accept_exec : @abi_decode_revert_6" });
    const accept_full_call = std.mem.indexOf(u8, main_fn, "icall @full") orelse return error.TestUnexpectedResult;
    const accept_full_label = std.mem.lastIndexOf(u8, main_fn[0..accept_full_call], "full_exec") orelse return error.TestUnexpectedResult;
    const accept_full_slice = main_fn[accept_full_label..accept_full_call];
    try testing.expect(std.mem.containsAtLeast(u8, accept_full_slice, 2, "calldataload"));
    try testing.expect(!std.mem.containsAtLeast(u8, accept_full_slice, 1, "signextend"));
    try testing.expect(!std.mem.containsAtLeast(u8, accept_full_slice, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, accept_full_slice, 1, "shl"));
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_3", "const 0x3", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_4", "const 0x4", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_5", "const 0x5", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_6", "const 0x6", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "accept_exec"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept a_accept b_accept"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept a_accept b_accept n_accept"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @accept a_accept b_accept n_accept arg_accept"));
}

test "compiler decodePermissive marker relaxes public calldata canonicality checks" {
    const source_text =
        \\contract Entry {
        \\    @decodePermissive
        \\    pub fn accept(flag: bool, owner: address, note: string) -> u256 {
        \\        if (flag) {
        \\            return note[0];
        \\        }
        \\        return @abiEncode(owner)[31];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const main_fn = try functionSlice(rendered, "main");

    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_4"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_5"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_11"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "mload8"));
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_14", "const 0xE", "mstore256", "revert" });
}

test "compiler abiDecode N3b4 validates public calldata enum range before call" {
    const source_text =
        \\enum Status: u8 { Active, Paused }
        \\contract Entry {
        \\    pub fn set(status: Status, flag: bool) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{ "a_set = calldataload", "and a_set", "eq a_set", "lt" });
    try expectOrderedNeedles(main_fn, &.{ "eq a_set", ": @abi_decode_revert_3" });
    try expectOrderedNeedles(main_fn, &.{ "lt", ": @abi_decode_revert_7" });
    try expectOrderedNeedles(main_fn, &.{ "b_set = calldataload", "eq b_set", "eq b_set", "or" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_3", "const 0x3", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_4", "const 0x4", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_7", "const 0x7", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "set_exec"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @set"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @set a_set b_set"));
}

test "compiler abiDecode N3b4 validates public calldata refinements before call" {
    const source_text =
        \\type PositiveByte = MinValue<u8, 1>;
        \\type SignedFloor = MinValue<i8, -5>;
        \\contract Entry {
        \\    pub fn check(amount: PositiveByte, delta: SignedFloor, owner: NonZeroAddress) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{ "a_check = calldataload", "and a_check", "eq a_check", "lt", "iszero", ": @abi_decode_revert_3", ": @abi_decode_revert_10" });
    try expectOrderedNeedles(main_fn, &.{ "b_check = calldataload", "signextend", "eq b_check", "slt", "iszero", ": @abi_decode_revert_3", ": @abi_decode_revert_10" });
    try expectOrderedNeedles(main_fn, &.{ "n_check = calldataload", "and n_check", "eq n_check", "eq", "iszero", ": @abi_decode_revert_5", ": @abi_decode_revert_10" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_3", "const 0x3", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_5", "const 0x5", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_10", "const 0xA", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "check_exec"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @check"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @check a_check b_check n_check"));
}

test "compiler abiDecode N3b4 validates public calldata string and bytes params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_text(text: string) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_data(data: bytes) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_both(text: string, data: bytes) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    for ([_][]const u8{ "take_text", "take_data" }) |name| {
        var load_buf: [64]u8 = undefined;
        var exec_buf: [64]u8 = undefined;
        const load = try std.fmt.bufPrint(&load_buf, "a_{s} = calldataload", .{name});
        const exec = try std.fmt.bufPrint(&exec_buf, "? @{s}_exec : @abi_decode_revert_11", .{name});
        try expectOrderedNeedles(main_fn, &.{
            load,
            "eq a_",
            ": @abi_decode_revert_11",
            "calldatasize",
            ": @abi_decode_revert_0",
            "calldataload",
            "large_const 0xFFFFFFFFFFFFFFFF",
            "gt",
            ": @abi_decode_revert_13",
            "const 0x100000",
            "gt",
            ": @abi_decode_revert_14",
            "div",
            "mul",
            ": @abi_decode_revert_0",
            "calldatacopy",
            "mload8",
            "eq",
            exec,
        });
    }
    try expectOrderedNeedles(main_fn, &.{
        "a_take_both = calldataload",
        "eq a_take_both",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_both = calldataload",
        "eq b_take_both",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x100000",
        "gt",
        ": @abi_decode_revert_14",
        "div",
        "mul",
        ": @abi_decode_revert_0",
        "calldatacopy",
        "mload8",
        "eq",
        "? @take_both_exec : @abi_decode_revert_11",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_14", "const 0xE", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_text"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_data"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_both"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_text a_take_text"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_data a_take_data"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_both_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata u256 slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_values(values: slice[u256]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_values(text: string, values: slice[u256]) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{
        "a_take_values = calldataload",
        "eq a_take_values",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @take_values_exec : @abi_decode_revert_0",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_values = calldataload",
        "eq a_take_text_values",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_values = calldataload",
        "eq b_take_text_values",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @take_text_values_exec : @abi_decode_revert_0",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_values_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_values_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata address slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_addresses(values: slice[address]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_addresses(text: string, values: slice[address]) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{
        "a_take_addresses = calldataload",
        "eq a_take_addresses",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "and",
        "eq",
        "? @take_addresses_exec : @abi_decode_revert_5",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_addresses = calldataload",
        "eq a_take_text_addresses",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_addresses = calldataload",
        "eq b_take_text_addresses",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "and",
        "eq",
        "? @take_text_addresses_exec : @abi_decode_revert_5",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_5", "const 0x5", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_addresses_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_addresses_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata bool slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_bools(values: slice[bool]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_bools(text: string, values: slice[bool]) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{
        "a_take_bools = calldataload",
        "eq a_take_bools",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "eq",
        "eq",
        "or",
        "? @take_bools_exec : @abi_decode_revert_4",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_bools = calldataload",
        "eq a_take_text_bools",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_bools = calldataload",
        "eq b_take_text_bools",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "eq",
        "eq",
        "or",
        "? @take_text_bools_exec : @abi_decode_revert_4",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_4", "const 0x4", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldatacopy"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_bools_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_bools_exec : @revert_error"));
}

test "compiler abiDecode N3b4 validates public calldata fixed bytes slice params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_tags(values: slice[bytes4]) -> bool {
        \\        return true;
        \\    }
        \\
        \\    pub fn take_text_tags(text: string, values: slice[bytes4]) -> bool {
        \\        return true;
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

    const main_fn = try functionSlice(rendered, "main");
    try expectOrderedNeedles(main_fn, &.{
        "a_take_tags = calldataload",
        "eq a_take_tags",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "shr",
        "shl",
        "eq",
        ": @abi_decode_revert_6",
    });
    try expectOrderedNeedles(main_fn, &.{
        "a_take_text_tags = calldataload",
        "eq a_take_text_tags",
        ": @abi_decode_revert_11",
        "calldatacopy",
        "mload8",
        "eq",
        "b_take_text_tags = calldataload",
        "eq b_take_text_tags",
        ": @abi_decode_revert_11",
        "calldatasize",
        ": @abi_decode_revert_0",
        "calldataload",
        "large_const 0xFFFFFFFFFFFFFFFF",
        "gt",
        ": @abi_decode_revert_13",
        "const 0x8000",
        "gt",
        ": @abi_decode_revert_9",
        "mul",
        "? @",
        ": @abi_decode_revert_0",
        "calldataload",
        "shr",
        "shl",
        "eq",
        ": @abi_decode_revert_6",
    });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_6", "const 0x6", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
    try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_tags"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @take_text_tags"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_tags_exec : @revert_error"));
    try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @take_text_tags_exec : @revert_error"));
}

test "compiler abiDecode N3b4 rejects unsupported nested dynamic calldata arrays before legacy fallback" {
    const source_text =
        \\contract Entry {
        \\    pub fn take_nested(values: slice[slice[u256]]) -> bool {
        \\        return true;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(!mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));
}

test "compiler abiDecode N3b4 validates public calldata dynamic tuple params before call" {
    const source_text =
        \\contract Entry {
        \\    pub fn value(t: (u256, string)) -> u256 {
        \\        return t.0 + t.1.len;
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

    const main_fn = try functionSlice(rendered, "main");
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "value_"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 2, "calldataload"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "abi_decode_revert_11"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "calldatacopy"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "mload8"));
    try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @value"));
}

test "compiler abiDecode N3b5 validates dynamic constructor string and bytes calldata" {
    const cases = [_][]const u8{
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(name: string) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(payload: bytes) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(name: string, payload: bytes) {
        \\        touched = 9;
        \\    }
        \\}
        ,
    };

    for (cases) |source_text| {
        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
        try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

        const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
        defer testing.allocator.free(rendered);

        const init_fn = try functionSlice(rendered, "init");
        try expectOrderedNeedles(init_fn, &.{ "codesize", "init_end", "codecopy", "mload256", "eq", ": @init_abi_decode_revert_11" });
        try expectOrderedNeedles(init_fn, &.{ "mload256", "gt", ": @init_abi_decode_revert_13" });
        try expectOrderedNeedles(init_fn, &.{ "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @init_abi_decode_revert_13", "const 0x100000", "gt", ": @init_abi_decode_revert_14", "div", "mul", ": @init_abi_decode_revert_0", "mload8" });
        try expectOrderedNeedles(init_fn, &.{ "mload8", "eq", "? @", ": @init_abi_decode_revert_11" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_14", "const 0xE", "mstore256", "revert" });
        try testing.expect(std.mem.containsAtLeast(u8, init_fn, 1, "icall @__ora_user_init"));
    }
}

test "compiler abiDecode N3b5 validates dynamic constructor slice calldata" {
    const cases = [_][]const u8{
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(values: slice[u256]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(owners: slice[address]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(flags: slice[bool]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(tags: slice[bytes4]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
        \\contract Entry {
        \\    storage var touched: u256 = 0;
        \\    pub fn init(name: string, owners: slice[address]) {
        \\        touched = 9;
        \\    }
        \\}
        ,
    };

    for (cases) |source_text| {
        var compilation = try compileText(source_text);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
        try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

        const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
        defer testing.allocator.free(rendered);

        const init_fn = try functionSlice(rendered, "init");
        try expectOrderedNeedles(init_fn, &.{ "codesize", "init_end", "codecopy", "mload256", "eq", ": @init_abi_decode_revert_11" });
        try expectOrderedNeedles(init_fn, &.{ "mload256", "gt", ": @init_abi_decode_revert_13" });
        try expectOrderedNeedles(init_fn, &.{ "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @init_abi_decode_revert_13", "const 0x8000", "gt", ": @init_abi_decode_revert_9", "mul", ": @init_abi_decode_revert_0" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_9", "const 0x9", "mstore256", "revert" });
        try expectOrderedNeedles(init_fn, &.{ "init_abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
        try testing.expect(std.mem.containsAtLeast(u8, init_fn, 1, "icall @__ora_user_init"));
    }
}

test "compiler abiDecode N3b5 validates dynamic Result carrier bounds before payload loads" {
    const cases = [_]struct {
        source: []const u8,
        cap: []const u8,
        cap_error: []const u8,
        payload_needles: []const []const u8,
    }{
        .{
            .source =
            \\error Failure(code: u256);
            \\
            \\contract Entry {
            \\    pub fn consume(value: Result<bytes, Failure>) -> u256 {
            \\        return match (value) {
            \\            Ok(inner) => inner.len,
            \\            Err(err) => err.code,
            \\        };
            \\    }
            \\}
            ,
            .cap = "const 0x100000",
            .cap_error = ": @abi_decode_revert_14",
            .payload_needles = &.{ ": @abi_decode_revert_0", "calldataload", "shr", "? @consume_exec : @abi_decode_revert_11" },
        },
        .{
            .source =
            \\error Failure(code: u256);
            \\
            \\contract Entry {
            \\    pub fn consume(value: Result<slice[address], Failure>) -> u256 {
            \\        return match (value) {
            \\            Ok(inner) => 1,
            \\            Err(err) => err.code,
            \\        };
            \\    }
            \\}
            ,
            .cap = "const 0x8000",
            .cap_error = ": @abi_decode_revert_9",
            .payload_needles = &.{ ": @abi_decode_revert_0", "calldataload", "and", "eq", "? @consume_exec : @abi_decode_revert_5" },
        },
    };

    for (cases) |case| {
        var compilation = try compileText(case.source);
        defer compilation.deinit();

        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
        try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));

        const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
        defer testing.allocator.free(rendered);

        const main_fn = try functionSlice(rendered, "main");
        try expectOrderedNeedles(main_fn, &.{ "calldatasize", ": @abi_decode_revert_0", "calldataload", "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @abi_decode_revert_13" });
        try expectOrderedNeedles(main_fn, &.{ "large_const 0xFFFFFFFFFFFFFFFF", "gt", ": @abi_decode_revert_13", case.cap, "gt", case.cap_error, "mul", ": @abi_decode_revert_0" });
        try expectOrderedNeedles(main_fn, case.payload_needles);
        try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_0", "const 0x0", "mstore256", "revert" });
        try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_11", "const 0xB", "mstore256", "revert" });
        try expectOrderedNeedles(main_fn, &.{ "abi_decode_revert_13", "const 0xD", "mstore256", "revert" });
        try testing.expect(std.mem.containsAtLeast(u8, main_fn, 1, "icall @consume"));
        try testing.expect(!std.mem.containsAtLeast(u8, main_fn, 1, "? @consume_exec : @revert_error"));
    }
}

test "compiler converts runtime abiDecode bool oversize priority through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bool_priority() -> u256 {
        \\        let decoded = @abiDecode(bool, hex"00000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000");
        \\        return match (decoded) {
        \\            Ok(_) => 99,
        \\            Err(_) => 1,
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
    const decode_fn = try functionSlice(rendered, "decode_bool_priority");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    // The oversize branch must be gated by the decoded Result tag so an earlier
    // decode error (invalid bool here) is not overwritten by oversize_buffer.
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "iszero"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "0x4"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "0x1"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode u8 memory result with canonical padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_u8(payload: bytes) -> u8 {
        \\        let decoded = @abiDecode(u8, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_u8:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_u8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode address memory result with canonical prefix validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_address(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(address, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_address:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_address");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bounds refinements with refinement validation" {
    const source_text =
        \\type PositiveAmount = MinValue<u256, 1>;
        \\type SmallAmount = MaxValue<u256, 10>;
        \\type RangedAmount = InRange<u256, 2, 8>;
        \\type PositiveByte = MinValue<u8, 1>;
        \\type SignedFloor = MinValue<i8, -5>;
        \\type SignedNonNegative = MinValue<i8, 0>;
        \\type NestedAmount = MinValue<MaxValue<u256, 10>, 1>;
        \\contract Decode {
        \\    pub fn decode_positive(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(PositiveAmount, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_small(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(SmallAmount, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_range(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(RangedAmount, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_positive_u8(payload: bytes) -> u8 {
        \\        let decoded = @abiDecode(PositiveByte, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_signed_i8(payload: bytes) -> i8 {
        \\        let decoded = @abiDecode(SignedFloor, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_signed_i8_zero(payload: bytes) -> i8 {
        \\        let decoded = @abiDecode(SignedNonNegative, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\    pub fn decode_nested(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(NestedAmount, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_positive");

    // Full-word u256 canonical decode emits no lt/gt comparison. Refined
    // targets now also get static length checks, so counts below distinguish
    // refinement comparisons from the length-check comparisons.
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));

    const max_decode_fn = try functionSlice(rendered, "decode_small");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(max_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, max_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, max_decode_fn, 1, "bitcast"));

    const range_decode_fn = try functionSlice(rendered, "decode_range");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(range_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, range_decode_fn, 1, "and"));
    try testing.expect(!std.mem.containsAtLeast(u8, range_decode_fn, 1, "bitcast"));

    const positive_u8_decode_fn = try functionSlice(rendered, "decode_positive_u8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(positive_u8_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, positive_u8_decode_fn, 1, "bitcast"));

    const signed_i8_decode_fn = try functionSlice(rendered, "decode_signed_i8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(signed_i8_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "slt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, signed_i8_decode_fn, 1, "bitcast"));

    const signed_i8_zero_decode_fn = try functionSlice(rendered, "decode_signed_i8_zero");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(signed_i8_zero_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "slt"));
    try testing.expect(std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, signed_i8_zero_decode_fn, 1, "bitcast"));

    const nested_decode_fn = try functionSlice(rendered, "decode_nested");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(nested_decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, nested_decode_fn, 1, "and"));
    try testing.expect(!std.mem.containsAtLeast(u8, nested_decode_fn, 1, "bitcast"));
}

test "compiler converts runtime abiDecode NonZeroAddress memory result with refinement validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_owner(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(NonZeroAddress, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_owner");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "iszero"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode i8 memory result with canonical sign extension" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_i8(payload: bytes) -> i8 {
        \\        let decoded = @abiDecode(i8, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_i8:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "signextend"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_i8");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode i256 memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_i256(payload: bytes) -> i256 {
        \\        let decoded = @abiDecode(i256, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_i256");

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_i256:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "signextend"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode fixed bytes memory result with canonical padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bytes4(payload: bytes) -> bytes4 {
        \\        let decoded = @abiDecode(bytes4, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => hex"00000000",
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_bytes4:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "eq"));
    const decode_fn = try functionSlice(rendered, "decode_bytes4");
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bytes1 memory result with canonical padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bytes1(payload: bytes) -> bytes1 {
        \\        let decoded = @abiDecode(bytes1, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => hex"00",
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
    const decode_fn = try functionSlice(rendered, "decode_bytes1");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bytes32 memory result without padding validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_bytes32(payload: bytes) -> bytes32 {
        \\        let decoded = @abiDecode(bytes32, payload);
        \\        return match (decoded) {
        \\            Ok(value) => value,
        \\            Err(_) => hex"0000000000000000000000000000000000000000000000000000000000000000",
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
    const decode_fn = try functionSlice(rendered, "decode_bytes32");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode enum memory result with range validation" {
    const source_text =
        \\enum Status: u8 { Active, Paused }
        \\contract Decode {
        \\    pub fn decode_status(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(Status, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_status");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode bitfield memory result with declared width validation" {
    const source_text =
        \\bitfield Flags: u8 {
        \\    enabled: bool @0;
        \\    mode: u7 @1;
        \\}
        \\contract Decode {
        \\    pub fn decode_flags(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(Flags, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_flags");

    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode void memory result with empty bytes validation" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_void(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(void, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_void");

    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode dynamic string and bytes memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_string(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(string, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_bytes(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(bytes, payload);
        \\        return match (decoded) {
        \\            Ok(_) => 1,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_values(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[u256], payload);
        \\        return match (decoded) {
        \\            Ok(values) => values[0] + values[1],
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_addresses(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[address], payload);
        \\        match (decoded) {
        \\            Ok(values) => {
        \\                if (values[0] == 0x0000000000000000000000000000000000000001) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_bools(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bool], payload);
        \\        match (decoded) {
        \\            Ok(values) => {
        \\                if (values[0]) {
        \\                    if (values[1]) {
        \\                        return 2;
        \\                    }
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_tags(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bytes4], payload);
        \\        match (decoded) {
        \\            Ok(tags) => {
        \\                const expected: bytes4 = hex"aabbccdd";
        \\                if (tags[0] == expected) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_tag_one(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bytes1], payload);
        \\        match (decoded) {
        \\            Ok(tags) => {
        \\                const expected: bytes1 = hex"aa";
        \\                if (tags[0] == expected) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_tag_full(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode(slice[bytes32], payload);
        \\        match (decoded) {
        \\            Ok(tags) => {
        \\                const expected: bytes32 = hex"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        \\                if (tags[0] == expected) {
        \\                    return 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_text(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, string), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0,
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_bytes(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, bytes), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0 + @cast(u256, value.1[0]),
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_values(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[u256]), payload);
        \\        return match (decoded) {
        \\            Ok(value) => value.0 + @cast(u256, value.1[0]) + value.1[1],
        \\            Err(_) => 0,
        \\        };
        \\    }
        \\
        \\    pub fn decode_pair_addresses(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[address]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                if (value.1[0] == 0x0000000000000000000000000000000000000001) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_bools(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bool]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                if (value.1[0]) {
        \\                    if (value.1[1]) {
        \\                        return value.0 + 2;
        \\                    }
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_tags(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bytes4]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                const expected: bytes4 = hex"aabbccdd";
        \\                if (value.1[0] == expected) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_tag_one(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bytes1]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                const expected: bytes1 = hex"aa";
        \\                if (value.1[0] == expected) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\
        \\    pub fn decode_pair_tag_full(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, slice[bytes32]), payload);
        \\        match (decoded) {
        \\            Ok(value) => {
        \\                const expected: bytes32 = hex"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        \\                if (value.1[0] == expected) {
        \\                    return value.0 + 1;
        \\                }
        \\                return 0;
        \\            }
        \\            Err(_) => {
        \\                return 0;
        \\            }
        \\        }
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);
    const string_fn = try functionSlice(rendered, "decode_string");
    const bytes_fn = try functionSlice(rendered, "decode_bytes");
    const values_fn = try functionSlice(rendered, "decode_values");
    const addresses_fn = try functionSlice(rendered, "decode_addresses");
    const bools_fn = try functionSlice(rendered, "decode_bools");
    const tags_fn = try functionSlice(rendered, "decode_tags");
    const tag_one_fn = try functionSlice(rendered, "decode_tag_one");
    const tag_full_fn = try functionSlice(rendered, "decode_tag_full");
    const pair_text_fn = try functionSlice(rendered, "decode_pair_text");
    const pair_bytes_fn = try functionSlice(rendered, "decode_pair_bytes");
    const pair_values_fn = try functionSlice(rendered, "decode_pair_values");
    const pair_addresses_fn = try functionSlice(rendered, "decode_pair_addresses");
    const pair_bools_fn = try functionSlice(rendered, "decode_pair_bools");
    const pair_tags_fn = try functionSlice(rendered, "decode_pair_tags");
    const pair_tag_one_fn = try functionSlice(rendered, "decode_pair_tag_one");
    const pair_tag_full_fn = try functionSlice(rendered, "decode_pair_tag_full");

    for ([_][]const u8{ string_fn, bytes_fn }) |decode_fn| {
        try expectDynamicAbiDecodeGuardChain(decode_fn);
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "mload256"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mload8"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "div"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "mul"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "sub"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 3, "lt"));
        try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "gt"));
        try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    }
    try expectDynamicAbiDecodeWordGuardChain(values_fn);
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 2, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 2, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, values_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, values_fn, 1, "mload8"));
    try testing.expect(!std.mem.containsAtLeast(u8, values_fn, 1, "bitcast"));
    try expectDynamicAbiDecodeWordGuardChain(addresses_fn);
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, addresses_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, addresses_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(bools_fn);
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, bools_fn, 1, "or"));
    try testing.expect(!std.mem.containsAtLeast(u8, bools_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(tags_fn);
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, tags_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, tags_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(tag_one_fn);
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_one_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_one_fn, 1, "mload8"));
    try expectDynamicAbiDecodeWordGuardChain(tag_full_fn);
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, tag_full_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_full_fn, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_full_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, tag_full_fn, 1, "mload8"));
    try expectDynamicAbiDecodeGuardChain(pair_text_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "mload8"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 1, "sub"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_text_fn, 2, "gt"));
    try expectMixedDynamicTupleCarrierShape(pair_text_fn);
    try expectDynamicAbiDecodeGuardChain(pair_bytes_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "mload8"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 1, "sub"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bytes_fn, 2, "gt"));
    try expectMixedDynamicTupleCarrierShape(pair_bytes_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_values_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 3, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 1, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 3, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_values_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_values_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_values_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_addresses_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_addresses_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_addresses_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_bools_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 1, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 1, "and"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_bools_fn, 1, "or"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_bools_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_bools_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_tags_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 2, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 4, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 1, "shl"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tags_fn, 1, "eq"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tags_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_tags_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_tag_one_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 2, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 4, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 2, "gt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 1, "shr"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_one_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_one_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_tag_one_fn);
    try expectDynamicAbiDecodeWordGuardChain(pair_tag_full_fn);
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 4, "mload256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 2, "malloc"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 4, "mstore256"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 2, "mul"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 4, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, pair_tag_full_fn, 2, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_full_fn, 1, "shr"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_full_fn, 1, "shl"));
    try testing.expect(!std.mem.containsAtLeast(u8, pair_tag_full_fn, 1, "mload8"));
    try expectMixedDynamicTupleCarrierShape(pair_tag_full_fn);
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
}

test "compiler converts runtime abiDecode u256 tuple memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_pair(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, u256), payload);
        \\        return match (decoded) {
        \\            Ok(pair) => pair.0 + pair.1,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_pair");

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "fn decode_pair:"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "const 0x20"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_create"));
}

test "compiler converts runtime abiDecode mixed static tuple memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_pair(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, bool), payload);
        \\        return match (decoded) {
        \\            Ok(pair) => pair.0,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_pair");

    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "or"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, decode_fn, 1, "bitcast"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_create"));
}

test "compiler converts runtime abiDecode nested static tuple memory result through OraToSIR" {
    const source_text =
        \\contract Decode {
        \\    pub fn decode_nested(payload: bytes) -> u256 {
        \\        let decoded = @abiDecode((u256, (bool, u256)), payload);
        \\        return match (decoded) {
        \\            Ok(pair) => pair.0 + pair.1.1,
        \\            Err(_) => 0,
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
    const decode_fn = try functionSlice(rendered, "decode_nested");

    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 3, "mload256"));
    try expectStaticAbiDecodeGuardBeforePayloadLoad(decode_fn);
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "const 0x20"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 2, "eq"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "or"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "lt"));
    try testing.expect(std.mem.containsAtLeast(u8, decode_fn, 1, "gt"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.tuple_create"));
}
