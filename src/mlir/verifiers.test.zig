const std = @import("std");
const testing = std.testing;
const c = @import("mlir_c_api").c;

const MlirContextHandle = struct {
    ctx: c.MlirContext,
};

fn createContext() MlirContextHandle {
    const ctx = c.oraContextCreate();
    const registry = c.oraDialectRegistryCreate();
    c.oraRegisterAllDialects(registry);
    c.oraContextAppendDialectRegistry(ctx, registry);
    c.oraDialectRegistryDestroy(registry);
    c.oraContextLoadAllAvailableDialects(ctx);
    _ = c.oraDialectRegister(ctx);
    return .{ .ctx = ctx };
}

fn destroyContext(handle: MlirContextHandle) void {
    c.oraContextDestroy(handle.ctx);
}

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

    const h = createContext();
    defer destroyContext(h);

    const module = parseModuleFromText(h.ctx, text) catch |err| switch (err) {
        error.MlirParseFailed => return,
        else => return err,
    };
    defer c.oraModuleDestroy(module);

    const verified = c.mlirOperationVerify(c.oraModuleGetOperation(module));
    try testing.expect(!verified);
}

test "ora ADT type parses, prints, and exposes its C API name" {
    const h = createContext();
    defer destroyContext(h);

    const variant_names = [_]c.MlirStringRef{
        c.oraStringRefCreate("Empty".ptr, "Empty".len),
        c.oraStringRefCreate("Value".ptr, "Value".len),
    };
    const payload_types = [_]c.MlirType{
        c.oraNoneTypeCreate(h.ctx),
        c.oraIntegerTypeGet(h.ctx, 256, false),
    };
    const adt_ty = c.oraAdtTypeGet(
        h.ctx,
        c.oraStringRefCreate("Event".ptr, "Event".len),
        variant_names.len,
        &variant_names,
        &payload_types,
    );
    try testing.expect(!c.oraTypeIsNull(adt_ty));
    try testing.expect(c.oraTypeIsAAdt(adt_ty));

    const name_ref = c.oraAdtTypeGetName(adt_ty);
    try testing.expectEqualStrings("Event", name_ref.data[0..name_ref.length]);
    try testing.expectEqual(@as(usize, 2), c.oraAdtTypeGetNumVariants(adt_ty));

    const second_variant_ref = c.oraAdtTypeGetVariantName(adt_ty, 1);
    try testing.expectEqualStrings("Value", second_variant_ref.data[0..second_variant_ref.length]);
    try testing.expect(c.mlirTypeEqual(payload_types[1], c.oraAdtTypeGetVariantPayloadType(adt_ty, 1)));

    const text =
        \\module {
        \\  func.func @f(%value: !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>)>) {
        \\    ora.return
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
}

test "ora ADT construct tag and payload ops verify declared variants" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @f(%payload: !ora.int<256, false>) -> !ora.int<256, false> {
        \\    %event = ora.adt.construct "Value"(%payload) : (!ora.int<256, false>) -> !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>)>
        \\    %tag = ora.adt.tag %event : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>)> -> !ora.int<256, false>
        \\    %extracted = ora.adt.payload %event, "Value" : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>)> -> !ora.int<256, false>
        \\    func.return %extracted : !ora.int<256, false>
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
}

test "ora ADT narrow construct tag and payload lower to SIR" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @f(%payload: !ora.int<128, false>) {
        \\    %event = ora.adt.construct "Value"(%payload) : (!ora.int<128, false>) -> !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<128, false>)>
        \\    %tag = ora.adt.tag %event : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<128, false>)> -> !ora.int<256, false>
        \\    %extracted = ora.adt.payload %event, "Value" : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<128, false>)> -> !ora.int<128, false>
        \\    ora.return
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
    try testing.expect(c.oraConvertToSIR(h.ctx, module, false));
}

test "mlir verifier rejects ora.adt.construct with wrong payload type" {
    const text =
        \\module {
        \\  func.func @f(%payload: !ora.bool) {
        \\    %event = ora.adt.construct "Value"(%payload) : (!ora.bool) -> !ora.adt<"Event", ("Value", !ora.int<256, false>)>
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
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

test "mlir verifier accepts resource boundary ops with map place paths" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @f(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operand_segment_sizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    "ora.create"(%balances, %to, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    "ora.destroy"(%balances, %from, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
}

test "mlir verifier accepts direct storage resource place roots" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.global "reserve" : !ora.int<256, false>
        \\  ora.global "treasury" : !ora.int<256, false>
        \\  func.func @f(%amount: !ora.int<256, false>) {
        \\    %reserve = "ora.sload"() <{global = "reserve"}> : () -> !ora.int<256, false>
        \\    %treasury = "ora.sload"() <{global = "treasury"}> : () -> !ora.int<256, false>
        \\    "ora.create"(%reserve, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>) -> ()
        \\    "ora.destroy"(%reserve, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>) -> ()
        \\    "ora.move"(%reserve, %treasury, %amount) <{operand_segment_sizes = array<i32: 1, 1, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
}

test "mlir verifier rejects copied values as resource places" {
    const text =
        \\module {
        \\  func.func @f(%source: !ora.int<256, false>, %destination: !ora.int<256, false>, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%source, %destination, %amount) <{operand_segment_sizes = array<i32: 1, 1, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "mlir verifier rejects resource ops without carrier_signed" {
    const text =
        \\module {
        \\  func.func @f(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operand_segment_sizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    try expectModuleVerificationFailure(text);
}

test "OraToSIR lowers resource create and destroy to checked storage updates" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module attributes {ora.global_slots_built, ora.global_slots = {balances = 0 : ui64}} {
        \\  ora.global "balances" : !ora.map<!ora.address, !ora.int<256, false>> {ora.slot_index = 0 : ui64}
        \\  func.func @f(%owner: !ora.address, %amount: !ora.int<256, false>) {
        \\    %balances = "ora.sload"() <{global = "balances"}> : () -> !ora.map<!ora.address, !ora.int<256, false>>
        \\    "ora.create"(%balances, %owner, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    "ora.destroy"(%balances, %owner, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    ora.return
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
    try testing.expect(c.oraConvertToSIR(h.ctx, module, false));
}

test "OraToSIR lowers resource move to alias-aware checked storage updates" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module attributes {ora.global_slots_built, ora.global_slots = {balances = 0 : ui64}} {
        \\  ora.global "balances" : !ora.map<!ora.address, !ora.int<256, false>> {ora.slot_index = 0 : ui64}
        \\  func.func @f(%from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    %balances = "ora.sload"() <{global = "balances"}> : () -> !ora.map<!ora.address, !ora.int<256, false>>
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operand_segment_sizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    ora.return
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
    try testing.expect(c.oraConvertToSIR(h.ctx, module, false));
}

test "OraToSIR fails closed for resource move without a storage root" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @f(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operand_segment_sizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    ora.return
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, text);
    defer c.oraModuleDestroy(module);

    try testing.expect(c.mlirOperationVerify(c.oraModuleGetOperation(module)));
    try testing.expect(!c.oraConvertToSIR(h.ctx, module, false));
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
