const common = @import("compiler.test.oratosir.common.zig");

const std = common.std;
const testing = common.testing;
const compiler = common.compiler;
const mlir = common.mlir;
const mlir_cfg = common.mlir_cfg;
const compileText = common.compileText;
const compilePackage = common.compilePackage;
const firstGuardIdFromModuleText = common.firstGuardIdFromModuleText;
const nthGuardIdFromModuleText = common.nthGuardIdFromModuleText;
const setAllRefinementGuardIds = common.setAllRefinementGuardIds;

test "compiler lowers guard clauses to runtime revert through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(amount: u256) -> bool
        \\        guard amount < 10;
        \\    {
        \\        return true;
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

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "sir.invalid"));
}

test "compiler lowers requires clauses to runtime revert and erases ensures before SIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(amount: u256) -> u256
        \\        requires amount < 10
        \\        ensures result == amount
        \\    {
        \\        return amount;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    {
        const before_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
        defer if (before_ref.data != null) mlir.oraStringRefFree(before_ref);
        const before = before_ref.data[0..before_ref.length];
        try testing.expect(std.mem.containsAtLeast(u8, before, 1, "ora.requires"));
        try testing.expect(std.mem.containsAtLeast(u8, before, 1, "ora.ensures"));
    }

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.requires"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.ensures"));
}

test "refinement cleanup keeps dominated requires checks without proven guard id" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value >= 0
        \\        requires value <= 200
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, after, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.requires"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.assert"));
}

test "refinement cleanup erases only proven guard while keeping dominated requires checks" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value >= 0
        \\        requires value <= 200
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    const before_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (before_ref.data != null) mlir.oraStringRefFree(before_ref);
    const before = before_ref.data[0..before_ref.length];
    const guard_id = try firstGuardIdFromModuleText(before);

    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer {
        var it = proven_guard_ids.iterator();
        while (it.next()) |entry| testing.allocator.free(entry.key_ptr.*);
        proven_guard_ids.deinit();
    }
    try proven_guard_ids.put(try testing.allocator.dupe(u8, guard_id), {});

    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, after, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.refinement_guard"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.requires"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.assert"));
}

test "refinement cleanup refuses duplicate guard ids even when proven" {
    const source_text =
        \\contract Check {
        \\    pub fn run(a: MinValue<u256, 1>, b: MinValue<u256, 1>) {
        \\        let _: u256 = a;
        \\        let _: u256 = b;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 2), setAllRefinementGuardIds(hir_result.context, hir_result.module.raw_module, "guard:duplicate"));

    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer {
        var it = proven_guard_ids.iterator();
        while (it.next()) |entry| testing.allocator.free(entry.key_ptr.*);
        proven_guard_ids.deinit();
    }
    try proven_guard_ids.put(try testing.allocator.dupe(u8, "guard:duplicate"), {});

    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, after, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.refinement_guard"));
}

test "refinement cleanup erases only matching distinct guard id" {
    const source_text =
        \\contract Check {
        \\    pub fn run(a: MinValue<u256, 1>, b: MinValue<u256, 1>) {
        \\        let _: u256 = a;
        \\        let _: u256 = b;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const before_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (before_ref.data != null) mlir.oraStringRefFree(before_ref);
    const before = before_ref.data[0..before_ref.length];
    const first_guard_id = try nthGuardIdFromModuleText(before, 0);
    _ = try nthGuardIdFromModuleText(before, 1);

    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer {
        var it = proven_guard_ids.iterator();
        while (it.next()) |entry| testing.allocator.free(entry.key_ptr.*);
        proven_guard_ids.deinit();
    }
    try proven_guard_ids.put(try testing.allocator.dupe(u8, first_guard_id), {});

    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, after, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.refinement_guard"));
}

test "Ora CFG overlay does not mark duplicate guard ids proven-erased" {
    const source_text =
        \\contract Check {
        \\    pub fn run(a: MinValue<u256, 1>, b: MinValue<u256, 1>) {
        \\        let _: u256 = a;
        \\        let _: u256 = b;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 2), setAllRefinementGuardIds(hir_result.context, hir_result.module.raw_module, "guard:duplicate"));

    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer {
        var it = proven_guard_ids.iterator();
        while (it.next()) |entry| testing.allocator.free(entry.key_ptr.*);
        proven_guard_ids.deinit();
    }
    try proven_guard_ids.put(try testing.allocator.dupe(u8, "guard:duplicate"), {});

    const dot = try mlir_cfg.generateCFG(hir_result.context, hir_result.module.raw_module, testing.allocator, .{
        .mode = .ora,
        .proven_guard_ids = &proven_guard_ids,
    });
    defer testing.allocator.free(dot);

    try testing.expect(std.mem.containsAtLeast(u8, dot, 1, "ora.refinement_guard"));
    try testing.expect(!std.mem.containsAtLeast(u8, dot, 1, "proof=\"proven-erased\""));
}

test "refinement duplicate helper treats null module as duplicated" {
    const null_module = mlir.MlirModule{ .ptr = null };
    try testing.expect(compiler.refinement_guards.guardIdIsDuplicated(null_module, "guard:null"));
}

test "refinement cleanup keeps dominated zero-address requires checks without proven guard id" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Check {
        \\    pub fn approve(spender: NonZeroAddress)
        \\        requires spender != std.constants.ZERO_ADDRESS
        \\        requires std.constants.ZERO_ADDRESS != spender
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, after, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.requires"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "ora.assert"));
}

test "source-inline refined arguments do not duplicate parameter refinement guards" {
    const source_text =
        \\contract Check {
        \\    inline fn requireOwner(owner: NonZeroAddress) {
        \\    }
        \\
        \\    pub fn refined(owner: NonZeroAddress) {
        \\        requireOwner(owner);
        \\    }
        \\
        \\    pub fn unrefined(owner: address) {
        \\        requireOwner(owner);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const before_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (before_ref.data != null) mlir.oraStringRefFree(before_ref);
    const before = before_ref.data[0..before_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, before, 1, "@requireOwner__inline__owner_refined_NonZeroAddress"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, before, "ora.refinement_guard"));
}

test "refinement cleanup keeps dominated requires checks in keep-proved mode" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value >= 0
        \\        requires value <= 200
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{
        .keep_proved_checks = true,
    });

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.count(u8, after, "cf.assert") > 1);
}

test "compiler lowers keep-proved requires checks through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value >= 0
        \\        requires value <= 200
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{
        .keep_proved_checks = true,
    });

    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "cf.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "sir.invalid"));
}

test "refinement cleanup preserves requires checks not implied by parameter refinement" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: InRange<u256, 0, 200>)
        \\        requires value <= 150
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, after, "cf.assert"));
}

test "refinement cleanup preserves assert message selector on cf.assert" {
    const source_text =
        \\contract Check {
        \\    pub fn run(flag: bool) {
        \\        assert(flag, "bad");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const selector = try compiler.hir.abi.keccakSelectorHex(testing.allocator, "bad");
    defer testing.allocator.free(selector);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "cf.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "ora.assert_selector"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, selector));
}

test "refinement cleanup tags kept refinement guards as clean runtime checks" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: MinValue<u256, 10>) {
        \\        let _: u256 = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var proven_guard_ids = std.StringHashMap(void).init(testing.allocator);
    defer proven_guard_ids.deinit();
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(hir_result.context, hir_result.module.raw_module, &proven_guard_ids, .{});

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "cf.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "ora.verification_type"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "guard"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "ora.verification_context"));
    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "refinement_guard"));
}

test "compiler lowers runtime refinement guards to clean revert through OraToSIR" {
    const source_text =
        \\contract Check {
        \\    pub fn run(value: MinValue<u256, 10>) {
        \\        let _: u256 = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const after = after_ref.data[0..after_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, after, 1, "sir.revert"));
    try testing.expect(!std.mem.containsAtLeast(u8, after, 1, "sir.invalid"));
}

test "compiler preserves requires source order before hazardous later clauses" {
    const source_text =
        \\contract Check {
        \\    pub fn ordered(a: u256, b: u256) -> u256
        \\        requires b != 0
        \\        requires a / b < 10
        \\    {
        \\        return a;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const after_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (after_ref.data != null) mlir.oraStringRefFree(after_ref);
    const rendered = after_ref.data[0..after_ref.length];

    const first_branch = std.mem.indexOf(u8, rendered, "sir.cond_br") orelse return error.TestUnexpectedResult;
    const division = std.mem.indexOf(u8, rendered, "sir.div") orelse return error.TestUnexpectedResult;
    try testing.expect(first_branch < division);
}

test "corpus guard runtime clause lowers to SIR revert" {
    var compilation = try compilePackage("ora-example/smt/verification/guard_runtime_clause.ora");
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir_result.module.raw_module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "sir.revert"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "guard_runtime_clause.ora"));
}
