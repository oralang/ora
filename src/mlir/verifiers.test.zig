const std = @import("std");
const testing = std.testing;
const mlir = @import("mod.zig");
const c = @import("mlir_c_api").c;

fn parseModuleFromText(ctx: c.MlirContext, text: []const u8) !c.MlirModule {
    const ref = c.oraStringRefCreate(text.ptr, text.len);
    const module = c.oraModuleCreateParse(ctx, ref);
    if (c.oraModuleIsNull(module)) {
        return error.MlirParseFailed;
    }
    return module;
}

fn expectModuleVerificationFailure(text: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const h = mlir.createContext(arena.allocator());
    defer mlir.destroyContext(h);

    const module = parseModuleFromText(h.ctx, text) catch |err| switch (err) {
        error.MlirParseFailed => return,
        else => return err,
    };
    defer c.oraModuleDestroy(module);

    const verified = c.mlirOperationVerify(c.oraModuleGetOperation(module));
    try testing.expect(!verified);
}

test "mlir verifier rejects ora.div with statically zero divisor" {
    const text =
        \\module {
        \\  func.func @f() {
        \\    %lhs = "arith.constant"() <{value = 42 : i256}> : () -> !ora.int<256, false>
        \\    %rhs = "arith.constant"() <{value = 0 : i256}> : () -> !ora.int<256, false>
        \\    %0 = "ora.div"(%lhs, %rhs) : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.rem with statically zero divisor" {
    const text =
        \\module {
        \\  func.func @f() {
        \\    %lhs = "arith.constant"() <{value = 42 : i256}> : () -> !ora.int<256, false>
        \\    %rhs = "arith.constant"() <{value = 0 : i256}> : () -> !ora.int<256, false>
        \\    %0 = "ora.rem"(%lhs, %rhs) : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.int<256, false>
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.sload with mismatched global type" {
    const text =
        \\module {
        \\  ora.contract @C {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @f() {
        \\      %0 = "ora.sload"() <{global = "counter"}> : () -> !ora.bool
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.sstore with mismatched global type" {
    const text =
        \\module {
        \\  ora.contract @C {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @f(%value: !ora.bool) {
        \\      "ora.sstore"(%value) <{global = "counter"}> : (!ora.bool) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.struct_init with wrong field operand type" {
    const text =
        \\module {
        \\  ora.contract @C {
        \\    "ora.struct.decl"() ({
        \\    }) {name = "Pair", sym_name = "Pair", ora.field_names = ["lhs", "rhs"], ora.field_types = [!ora.int<256, false>, !ora.bool]} : () -> ()
        \\    func.func @f() {
        \\      %lhs = "arith.constant"() <{value = 1 : i256}> : () -> !ora.int<256, false>
        \\      %rhs = "arith.constant"() <{value = 2 : i256}> : () -> !ora.int<256, false>
        \\      %0 = "ora.struct_init"(%lhs, %rhs) : (!ora.int<256, false>, !ora.int<256, false>) -> !ora.struct<"Pair">
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.struct_init with wrong field count" {
    const text =
        \\module {
        \\  ora.contract @C {
        \\    "ora.struct.decl"() ({
        \\    }) {name = "Pair", sym_name = "Pair", ora.field_names = ["lhs", "rhs"], ora.field_types = [!ora.int<256, false>, !ora.bool]} : () -> ()
        \\    func.func @f() {
        \\      %lhs = "arith.constant"() <{value = 1 : i256}> : () -> !ora.int<256, false>
        \\      %0 = "ora.struct_init"(%lhs) : (!ora.int<256, false>) -> !ora.struct<"Pair">
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.map_get with wrong key type" {
    const text =
        \\module {
        \\  func.func @f(%map: !ora.map<!ora.int<256, false>, !ora.int<256, false>>, %key: !ora.bool) {
        \\    %0 = "ora.map_get"(%map, %key) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.bool) -> !ora.int<256, false>
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.map_store with wrong key type" {
    const text =
        \\module {
        \\  func.func @f(%map: !ora.map<!ora.int<256, false>, !ora.int<256, false>>, %key: !ora.bool, %value: !ora.int<256, false>) {
        \\    "ora.map_store"(%map, %key, %value) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.bool, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.error.unwrap on non-error-union type" {
    const text =
        \\module {
        \\  func.func @f(%value: !ora.int<256, false>) {
        \\    %0 = "ora.error.unwrap"(%value) : (!ora.int<256, false>) -> !ora.int<256, false>
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.error.unwrap with mismatched result type" {
    const text =
        \\module {
        \\  func.func @f(%value: !ora.error_union<!ora.int<256, false>>) {
        \\    %0 = "ora.error.unwrap"(%value) : (!ora.error_union<!ora.int<256, false>>) -> !ora.bool
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.return with wrong operand count" {
    const text =
        \\module {
        \\  func.func @f() -> !ora.int<256, false> {
        \\    ora.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.return with wrong operand type" {
    const text =
        \\module {
        \\  func.func @f(%value: !ora.bool) -> !ora.int<256, false> {
        \\    ora.return %value : !ora.bool
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.refinement_to_base with inconsistent base type" {
    const text =
        \\module {
        \\  func.func @f(%value: !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>) {
        \\    %0 = "ora.refinement_to_base"(%value) : (!ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>) -> !ora.bool
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.base_to_refinement with inconsistent base type" {
    const text =
        \\module {
        \\  func.func @f(%value: !ora.bool) {
        \\    %0 = "ora.base_to_refinement"(%value) : (!ora.bool) -> !ora.min_value<!ora.int<256, false>, 0, 0, 0, 1>
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.switch_expr with empty case region" {
    const text =
        \\module {
        \\  func.func @choose(%tag: i256) -> i256 {
        \\    %0 = ora.switch_expr %tag : i256 -> i256 {
        \\      case 3 => {}
        \\      else => {
        \\        %1 = arith.constant 9 : i256
        \\        ora.yield %1 : i256
        \\      }
        \\    }
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects ora.switch_expr with mismatched metadata counts" {
    const text =
        \\module {
        \\  func.func @choose(%tag: i256) -> i256 {
        \\    %0 = "ora.switch_expr"(%tag) <{case_values = array<i64: 3>, range_starts = array<i64: 0>, range_ends = array<i64: 0>, case_kinds = array<i64: 0>, default_case_index = 1 : i64}> ({
        \\      %1 = arith.constant 7 : i256
        \\      ora.yield %1 : i256
        \\    }, {
        \\      %2 = arith.constant 9 : i256
        \\      ora.yield %2 : i256
        \\    }) : (i256) -> i256
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}
