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

test "compiler rejects invalid concrete generic arithmetic instantiations during typecheck" {
    const source_text =
        \\contract Test {
        \\    fn add(comptime T: type, a: T, b: T) -> T {
        \\        return a + b;
        \\    }
        \\
        \\    pub fn run(a: address, b: address) -> address {
        \\        return add(address, a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), typecheck.diagnostics.items.items.len);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "invalid binary operator '+' for types 'address' and 'address'"));
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

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "call @ct") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "arith.constant 18 : i8") != null);
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

test "compiler parses top-level comptime functions" {
    const source_text =
        \\comptime fn helper() -> u256 {
        \\    return 1;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return helper();
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const file = try compilation.db.astFile(module.file_id);

    try testing.expectEqual(@as(usize, 2), file.root_items.len);

    const helper = file.item(file.root_items[0]).Function;
    try testing.expectEqualStrings("helper", helper.name);
    try testing.expect(helper.is_comptime);

    const run = file.item(file.root_items[1]).Function;
    try testing.expectEqualStrings("run", run.name);
    try testing.expect(!run.is_comptime);
}

test "compiler allows generic arithmetic against concrete integer constants in template bodies" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\fn add(comptime T: type, a: T, b: T) -> T
        \\    requires a <= std.constants.U256_MAX - b
        \\{
        \\    return a + b;
        \\}
        \\
        \\pub fn run(a: u256, b: u256) -> u256 {
        \\    return add(u256, a, b);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
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

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.constant 5 : i256"));
}

test "compiler folds comptime shifts before MLIR verification" {
    const source_text =
        \\contract ComptimeShiftProbe {
        \\    pub fn test_large_shr() -> u256 {
        \\        const x: u256 = 1 >> 300;
        \\        const y: u256 = 255 >> 256;
        \\        const z: u256 = 1 >> 8;
        \\        const w: u256 = 256 >> 4;
        \\        return x + y + z + w;
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "arith.shrui"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "arith.addi"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.constant 16 : i256"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.return"));
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

test "compiler lowers overflow asserts through private generic checked helpers" {
    const source_text =
        \\contract FailPrivateGenericCheckedArith {
        \\    fn add(comptime T: type, a: T, b: T) -> T {
        \\        return a + b;
        \\    }
        \\
        \\    pub fn uncheckedAdd(a: u256, b: u256) -> u256 {
        \\        return add(u256, a, b);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @add__u256"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.addi"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
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

test "compiler const eval rejects invalid refinement struct field construction" {
    const source_text =
        \\struct Box {
        \\    value: MinValue<u256, 10>,
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let box = Box { value: 5 };
        \\        box.value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&consteval.diagnostics, "comptime refinement violation"));
    try testing.expect(diagnosticMessagesContain(&consteval.diagnostics, "expected MinValue value >= 10"));
}

test "compiler const eval rejects invalid refinement ADT payload construction" {
    const source_text =
        \\enum MaybeAmount {
        \\    None,
        \\    Value(MinValue<u256, 10>),
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let maybe = MaybeAmount.Value(5);
        \\        0;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&consteval.diagnostics, "comptime refinement violation"));
    try testing.expect(diagnosticMessagesContain(&consteval.diagnostics, "expected MinValue value >= 10"));
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
    try testing.expectEqual(@as(usize, 0), consteval_diags.items.items.len);
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

test "compiler const eval reports explicit iteration-limit diagnostics for comptime for loops" {
    const source_text =
        \\pub fn sum() -> u256 {
        \\    return comptime {
        \\        let total = 0;
        \\        for (5) |i| {
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
    var consteval = try compiler.comptime_eval.constEval(testing.allocator, ast_file, .{
        .config = .{
            .max_loop_iterations = 2,
        },
    });
    defer consteval.deinit();

    try testing.expectEqual(@as(usize, 1), consteval.diagnostics.items.items.len);
    const diag = consteval.diagnostics.items.items[0];
    try testing.expect(std.mem.indexOf(u8, diag.message, "comptime loop iteration limit exceeded") != null);
    try testing.expect(std.mem.indexOf(u8, diag.message, "evaluation exceeded max_loop_iterations") != null);
}

test "compiler raw const eval leaves recursive comptime call unresolved under low recursion cap" {
    const source_text =
        \\comptime fn factorial(n: u256) -> u256 {
        \\    if (n == 0) { return 1; }
        \\    return n * factorial(n - 1);
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        factorial(100);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    var consteval = try compiler.comptime_eval.constEval(testing.allocator, ast_file, .{
        .config = .{
            .max_recursion_depth = 8,
            .max_steps = 1000,
            .max_loop_iterations = 1000,
        },
    });
    defer consteval.deinit();

    try testing.expect(consteval.diagnostics.items.items.len >= 1);
}

test "compiler const eval reports explicit step-limit diagnostics" {
    const source_text =
        \\pub fn sum() -> u256 {
        \\    return comptime {
        \\        let total = 0;
        \\        for (5) |i| {
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
    var consteval = try compiler.comptime_eval.constEval(testing.allocator, ast_file, .{
        .config = .{
            .max_loop_iterations = 100,
            .max_steps = 8,
        },
    });
    defer consteval.deinit();

    try testing.expect(consteval.diagnostics.items.items.len >= 1);
    try testing.expect(diagnosticMessagesContain(&consteval.diagnostics, "evaluation step limit exceeded"));
    try testing.expect(diagnosticMessagesContain(&consteval.diagnostics, "evaluation exceeded max_steps"));
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

test "compiler const eval mutates nested struct fields by declared field id" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\struct Holder {
        \\    pair: Pair;
        \\    marker: u256;
        \\}
        \\
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let holder = Holder {
        \\            marker: 100,
        \\            pair: Pair { second: 2, first: 1 },
        \\        };
        \\        holder.pair.second += 40;
        \\        holder.pair.first + holder.pair.second + holder.marker;
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
    try testing.expectEqual(@as(i128, 143), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval reads string length and indexing" {
    const source_text =
        \\pub fn get() -> u256 {
        \\    return comptime {
        \\        let text = "hello";
        \\        text.len + text[1];
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

test "compiler const eval matches payload-carrying enum variants" {
    const source_text =
        \\enum Event {
        \\    Empty,
        \\    Value(u256),
        \\    Pair(u256, u256),
        \\}
        \\
        \\pub fn folded() -> u256 {
        \\    return comptime {
        \\        let event = Event.Pair(2, 3);
        \\        let folded = switch (event) {
        \\            Event.Empty => 0,
        \\            Event.Value(value) => value,
        \\            Event.Pair(lhs, rhs) => lhs + rhs,
        \\        };
        \\        folded;
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

test "compiler const eval folds pure helper calls in ordinary const bindings" {
    const source_text =
        \\fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    const val: u256 = add(2, 3);
        \\    return val;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[decl.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval folds pure loop helper calls in ordinary const bindings" {
    const source_text =
        \\fn power(base: u256, exp: u256) -> u256 {
        \\    var acc: u256 = 1;
        \\    var i: u256 = 0;
        \\    while (i < exp) {
        \\        acc = acc * base;
        \\        i = i + 1;
        \\    }
        \\    return acc;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    const val: u256 = power(2, 10);
        \\    return val;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 1024), try consteval.values[decl.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval folds top-level comptime loop helpers and skips runtime lowering" {
    const source_text =
        \\comptime fn power(base: u256, exp: u256) -> u256 {
        \\    var acc: u256 = 1;
        \\    var i: u256 = 0;
        \\    while (i < exp) {
        \\        acc = acc * base;
        \\        i = i + 1;
        \\    }
        \\    return acc;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    const val: u256 = power(2, 10);
        \\    return val;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 1024), try consteval.values[decl.value.?.index()].?.integer.toInt(i128));

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "func.func @power"));
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @power"));
    try testing.expect(std.mem.indexOf(u8, rendered, " 1024 : i256") != null);
}

test "compiler const eval folds top-level comptime for-loop helpers and skips runtime lowering" {
    const source_text =
        \\comptime fn sum_first_n(n: u256) -> u256 {
        \\    var total: u256 = 0;
        \\    for (n) |i| {
        \\        total = total + i;
        \\    }
        \\    return total;
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    const val: u256 = sum_first_n(5);
        \\    return val;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 10), try consteval.values[decl.value.?.index()].?.integer.toInt(i128));

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "func.func @sum_first_n"));
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @sum_first_n"));
    try testing.expect(std.mem.indexOf(u8, rendered, " 10 : i256") != null);
}

test "compiler const eval folds pure recursive helper calls in ordinary const bindings" {
    const source_text =
        \\comptime fn factorial(n: u256) -> u256 {
        \\    if (n == 0) { return 1; }
        \\    return n * factorial(n - 1);
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    const val: u256 = factorial(5);
        \\    return val;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 120), try consteval.values[decl.value.?.index()].?.integer.toInt(i128));

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "func.func @factorial"));
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @factorial"));
    try testing.expect(std.mem.indexOf(u8, rendered, " 120 : i256") != null);
}

test "compiler const eval folds contract member helper calls in ordinary const bindings" {
    const source_text =
        \\contract Sample {
        \\    fn add(a: u256, b: u256) -> u256 {
        \\        return a + b;
        \\    }
        \\
        \\    fn power(base: u256, exp: u256) -> u256 {
        \\        var acc: u256 = 1;
        \\        var i: u256 = 0;
        \\        while (i < exp) {
        \\            acc = acc * base;
        \\            i = i + 1;
        \\        }
        \\        return acc;
        \\    }
        \\
        \\    pub fn run() -> u256 {
        \\        const a: u256 = add(2, 3);
        \\        const b: u256 = power(2, 10);
        \\        return a + b;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const function = ast_file.item(contract.members[2]).Function;
    const body = ast_file.body(function.body);
    const decl_a = ast_file.statement(body.statements[0]).VariableDecl;
    const decl_b = ast_file.statement(body.statements[1]).VariableDecl;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(i128, 5), try consteval.values[decl_a.value.?.index()].?.integer.toInt(i128));
    try testing.expectEqual(@as(i128, 1024), try consteval.values[decl_b.value.?.index()].?.integer.toInt(i128));

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @add"));
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @power"));
    try testing.expect(std.mem.indexOf(u8, rendered, "arith.constant 1029") != null);
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

test "compiler const eval matches payload enum values" {
    const source_text =
        \\enum Choice {
        \\    Empty,
        \\    Value(u256),
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let choice = Choice.Value(41);
        \\        let out = match (choice) {
        \\            Choice.Empty => 0,
        \\            Choice.Value(value) => value + 1,
        \\        };
        \\        out;
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
    try testing.expectEqual(@as(i128, 42), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler const eval matches Result error variants" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\comptime fn choose(flag: bool) -> Result<u256, Failure> {
        \\    if (flag) {
        \\        return Ok(7);
        \\    }
        \\    return Err(Failure(9));
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let maybe = choose(false);
        \\        let out = match (maybe) {
        \\            Ok(value) => value,
        \\            Failure(code) => code + 1,
        \\        };
        \\        out;
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
    try testing.expectEqual(@as(i128, 10), try consteval.values[ret_stmt.value.?.index()].?.integer.toInt(i128));
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

test "compiler surfaces comptime stage diagnostics through db" {
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
    try testing.expect(diagnosticMessagesContain(consteval_diags, "comptime block did not produce a value"));

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const typecheck_diags = &module_typecheck.diagnostics;
    try testing.expectEqual(@as(usize, 0), typecheck_diags.len());
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

test "comptime constant if dead branch is removed before SIR emission" {
    var compilation = try compilePackage("ora-example/debugger/comptime_debug_probe.ora");
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.const 999"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "removedPathMarker"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora-example/debugger/comptime_debug_probe.ora\":37"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora-example/debugger/comptime_debug_probe.ora\":38"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora-example/debugger/comptime_debug_probe.ora\":39"));
}

test "generic struct field extract checked add verifies without degradation" {
    const path = "ora-example/comptime/generics/generic_struct_multi_type_params.ora";
    const function_name = "value_plus_one";

    var baseline = try verifyExampleWithoutDegradation(path, function_name, false, 5_000);
    defer baseline.deinit(testing.allocator);

    try testing.expect(!baseline.degraded);
    try testing.expect(baseline.success);
    try testing.expectEqual(@as(usize, 0), baseline.errors_len);

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var rerun = try verifyExampleWithoutDegradation(path, function_name, false, 5_000);
        defer rerun.deinit(testing.allocator);

        try expectVerificationProbeEquivalent(&baseline, &rerun);
        try testing.expect(!rerun.degraded);
        try testing.expect(rerun.success);
        try testing.expectEqual(@as(usize, 0), rerun.errors_len);
    }
}
