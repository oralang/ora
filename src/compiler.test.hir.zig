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

test "compiler lowers impl self methods and calls end to end" {
    const source_text =
        \\trait CounterLike {
        \\    fn get(self) -> u256;
        \\    fn bump(self, amount: u256) -> u256;
        \\}
        \\
        \\struct Counter {
        \\    total: u256,
        \\}
        \\
        \\impl CounterLike for Counter {
        \\    fn get(self) -> u256 { return self.total; }
        \\    fn bump(self, amount: u256) -> u256 {
        \\        return self.get() + amount;
        \\    }
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    let c: Counter = .{ .total = 0 };
        \\    return c.bump(1);
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @Counter.bump"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "call @Counter.bump"));
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

test "compiler lowers embedded std constants through imported module access" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\pub fn max_value() -> u256 {
        \\    return std.constants.U256_MAX;
        \\}
        \\
        \\pub fn zero_address() -> address {
        \\    return std.constants.ZERO_ADDRESS;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.indexOf(u8, hir_text, "ora.field_access") == null);
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

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum_constant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum_ordinal"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
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

test "compiler lowers tuple top-level const items through ora.const" {
    const source_text =
        \\const PAIR: (u256, u256) = @divmod(17, 5);
        \\
        \\fn run() -> u256 {
        \\    return PAIR.0 * 5 + PAIR.1;
        \\}
    ;

    const ora_text = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(ora_text);

    try testing.expect(std.mem.containsAtLeast(u8, ora_text, 1, "ora.const"));
    try testing.expect(std.mem.containsAtLeast(u8, ora_text, 1, "PAIR"));
    try testing.expect(std.mem.containsAtLeast(u8, ora_text, 1, "["));
    try testing.expect(!std.mem.containsAtLeast(u8, ora_text, 1, "ora.constant_decl"));
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shrui"));
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

test "compiler lowers keyword logical operators with short-circuit control flow" {
    const source_text =
        \\comptime const std = @import("std");
        \\contract Probe {
        \\    pub fn run(ts: u256) -> bool {
        \\        return ts > 0 and std.block.timestamp() > ts or ts == 7;
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "arith.andi"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "arith.ori"));
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

test "compiler lowers divExact and divmod through Ora and SIR" {
    const source_text =
        \\pub fn compute(a: i256, b: i256, x: u256, y: u256) -> i256 {
        \\    let exact = @divExact(12, 4);
        \\    let rounded = @divFloor(a, b);
        \\    let pair = @divmod(x, y);
        \\    return rounded + exact + @cast(i256, pair.0) + @cast(i256, pair.1);
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.assert") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.tuple_create") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.tuple_extract") != null);

    var compilation = try compileText(source_text);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.shrui"));
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.divui"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.remui"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
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
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "arith.constant 9 : i256"));
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

test "compiler lowers native string and bytes len field access" {
    const source_text =
        \\pub fn string_len(text: string) -> u256 {
        \\    return text.len;
        \\}
        \\
        \\pub fn bytes_len(data: bytes) -> u256 {
        \\    return data.len;
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "ora.length"));
}

test "compiler lowers checked cast of native bytes length through asserted truncation" {
    const source_text =
        \\pub fn bytes_len_checked(data: bytes) -> u8 {
        \\    return @cast(u8, data.len);
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.length"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "arith.trunci"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "safe cast narrowing overflow"));
}

test "compiler lowers explicit truncate of native bytes length without overflow assertion" {
    const source_text =
        \\pub fn bytes_len_truncated(data: bytes) -> u8 {
        \\    return @truncate(u8, data.len);
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.length"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "arith.trunci"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "safe cast narrowing overflow"));
}

test "compiler lowers embedded std bytes helpers through imported module access" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\pub fn empty(data: bytes) -> bool {
        \\    return std.bytes.isEmpty(data);
        \\}
        \\
        \\pub fn same(a: bytes, b: bytes) -> bool {
        \\    return std.bytes.eq(a, b);
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "call @std.bytes.isEmpty"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @std.bytes.isEmpty"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "call @std.bytes.eq"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @std.bytes.eq"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "ora.byte_at"));
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

