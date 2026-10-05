//! Fresh, non-authorizing inventory. Submitted rows are compared to live HIR;
//! effects and bodies are recorded, not declared semantically verified.
const std = @import("std");
const mlir = @import("mlir_c_api").c;
const z3 = @import("ora_z3_verification");
const builtin = @import("ora_types").builtin;
const data = @import("storage_lifecycle.zig");
const canonical = @import("obligation_from_mlir.zig");
const lean = @import("obligation_to_lean.zig");

pub const Snapshot = struct {
    arena: std.heap.ArenaAllocator,
    inventory: data.Inventory,
    function_ops: []const mlir.MlirOperation,

    pub fn deinit(self: *Snapshot) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn collect(allocator: std.mem.Allocator, module: mlir.MlirModule, pass: *z3.VerificationPass) !Snapshot {
    if (mlir.oraModuleIsNull(module)) return error.MissingLifecycleModule;
    const selected = try pass.lifecycleRuntimeFunctions(allocator, module);
    defer allocator.free(selected);
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    var collector: Collector = .{ .allocator = arena.allocator(), .selected = selected };
    const root = mlir.oraModuleGetOperation(module);
    try collector.walk(root, "", null);
    for (collector.functions.items) |*function| {
        for (@constCast(function.events)) |*event| {
            if (event.kind != .call or event.refusal != null) continue;
            const target = event.target orelse {
                event.refusal = .unresolved_call;
                continue;
            };
            const resolved = for (collector.functions.items) |callee| {
                if (std.mem.eql(u8, callee.symbol, target)) break true;
            } else false;
            if (!resolved) event.refusal = .unresolved_call;
        }
    }
    const inventory: data.Inventory = .{
        .module_sha256 = try operationHash(root),
        .claims = collector.claims.items,
        .functions = collector.functions.items,
    };
    try inventory.validate();
    return .{ .arena = arena, .inventory = inventory, .function_ops = collector.function_ops.items };
}

pub const Projection = struct {
    snapshot: Snapshot,
    canonical: canonical.CollectResult,
    bindings: []const data.FunctionBinding,

    pub fn deinit(self: *Projection) void {
        self.canonical.deinit();
        self.snapshot.deinit();
        self.* = undefined;
    }
};

/// Complete fresh roots or a refusal. This entry point has no CLI consumer.
pub fn project(allocator: std.mem.Allocator, module: mlir.MlirModule, pass: *z3.VerificationPass, expected: data.Inventory) !Projection {
    var snapshot = try collect(allocator, module, pass);
    errdefer snapshot.deinit();
    if (!std.mem.eql(u8, &(try expected.sha256(allocator)), &(try snapshot.inventory.sha256(allocator)))) return error.LifecycleInventoryMismatch;
    const arena = snapshot.arena.allocator();
    var roots: std.ArrayList(mlir.MlirOperation) = .empty;
    for (snapshot.inventory.functions, snapshot.function_ops) |function, op| {
        if (function.runtime or function.constructor) try roots.append(arena, op);
    }
    var rows = try canonical.collect(allocator, module, .{ .project_calls = true, .lifecycle_roots = roots.items });
    errdefer rows.deinit();
    if (rows.set.hasBlockingDiagnostic()) return error.LifecycleProjectionUnsupported;
    const bindings = try arena.alloc(data.FunctionBinding, rows.set.function_summaries.len);
    for (rows.set.function_summaries, rows.function_ops, bindings) |row, op, *binding| {
        const index = for (snapshot.function_ops, 0..) |candidate, i| {
            if (@intFromPtr(candidate.ptr) == op) break i;
        } else return error.ForeignLifecycleBinding;
        binding.* = .{ .function_ordinal = snapshot.inventory.functions[index].ordinal, .function_id = row.id, .function_vc_sha256 = try lean.functionVcSha256(rows.set, row.id) };
    }
    try data.validateBindings(snapshot.inventory, bindings);
    try checkFresh(allocator, module, pass, snapshot.inventory);
    return .{ .snapshot = snapshot, .canonical = rows, .bindings = bindings };
}

pub fn checkFresh(allocator: std.mem.Allocator, module: mlir.MlirModule, pass: *z3.VerificationPass, expected: data.Inventory) !void {
    var fresh = try collect(allocator, module, pass);
    defer fresh.deinit();
    if (!std.mem.eql(u8, &(try expected.sha256(allocator)), &(try fresh.inventory.sha256(allocator)))) return error.LifecycleInventoryMismatch;
}

const Collector = struct {
    allocator: std.mem.Allocator,
    selected: []const mlir.MlirOperation,
    claims: std.ArrayList(data.Claim) = .empty,
    functions: std.ArrayList(data.Function) = .empty,
    function_ops: std.ArrayList(mlir.MlirOperation) = .empty,
    ordinal: u32 = 0,

    fn string(self: *Collector, op: mlir.MlirOperation, name: []const u8) !?[]const u8 {
        const value = attr(op, name);
        if (mlir.oraAttributeIsNull(value) or !mlir.mlirAttributeIsAString(value)) return null;
        const text = mlir.oraStringAttrGetValue(value);
        if (text.data == null) return null;
        return try self.allocator.dupe(u8, text.data[0..text.length]);
    }

    fn claim(self: *Collector, op: mlir.MlirOperation, owner: []const u8, ordinal: u32) !void {
        const type_attr = attr(op, "type");
        if (mlir.oraAttributeIsNull(type_attr) or !mlir.mlirAttributeIsAType(type_attr)) return error.InvalidLifecycleType;
        const ty = mlir.oraTypeAttrGetValue(type_attr);
        if (mlir.oraTypeIsNull(ty)) return error.InvalidLifecycleType;
        if (mlir.oraTypeIsNull(mlir.oraRefinementTypeGetBaseType(ty))) return;
        var row: data.Claim = .{ .owner = owner, .root = (try self.string(op, "sym_name")) orelse return error.InvalidLifecycleOwner, .ordinal = ordinal, .type_id = null, .bounds = null, .zero_base = false, .refusal = null };
        const id_attr = attr(op, "ora.compiler_type_id");
        if (!mlir.oraAttributeIsNull(id_attr) and mlir.mlirAttributeIsAInteger(id_attr)) row.type_id = std.math.cast(u32, try integer(id_attr));
        var kind: u32 = 0;
        var width: u32 = 0;
        var minimum: [4]u64 = undefined;
        var maximum: [4]u64 = undefined;
        if (!mlir.oraRefinementTypeGetIntegerBounds(ty, &kind, &width, &minimum, &maximum)) {
            row.refusal = .unsupported_type;
        } else if (builtin.integerInfoByComptimeTypeId(row.type_id orelse 0)) |info| {
            if (info.width != width) row.refusal = .width_mismatch else row.bounds = .{
                .kind = switch (kind) {
                    1 => .MinValue,
                    2 => .MaxValue,
                    3 => .InRange,
                    else => return error.InvalidLifecycleBounds,
                },
                .minimum = limbs(minimum),
                .maximum = limbs(maximum),
            };
        } else row.refusal = .missing_type;
        const initial = attr(op, "init");
        // Globals are declarations; deployment starts with EVM zero storage.
        // A nonzero/unknown attribute is not treated as an executed initializer.
        row.zero_base = !mlir.oraAttributeIsNull(initial) and (mlir.mlirAttributeIsAUnit(initial) or
            (mlir.mlirAttributeIsAInteger(initial) and try integer(initial) == 0));
        if (!row.zero_base and row.refusal == null) row.refusal = .unsupported_initialization;
        try self.claims.append(self.allocator, row);
    }

    fn walk(self: *Collector, op: mlir.MlirOperation, inherited_owner: []const u8, inherited_function: ?usize) !void {
        const ordinal = self.ordinal;
        self.ordinal = try std.math.add(u32, self.ordinal, 1);
        const name_ref = mlir.oraOperationGetName(op);
        defer if (name_ref.data != null) mlir.oraStringRefFree(name_ref);
        if (name_ref.data == null) return error.MissingLifecycleOperation;
        const name = name_ref.data[0..name_ref.length];
        var owner = inherited_owner;
        var function = inherited_function;
        if (std.mem.eql(u8, name, "ora.contract")) owner = try std.fmt.allocPrint(self.allocator, "module/contract:{s}", .{(try self.string(op, "sym_name")) orelse return error.InvalidLifecycleOwner});
        if (std.mem.eql(u8, name, "ora.global")) try self.claim(op, owner, ordinal);
        if (std.mem.eql(u8, name, "func.func")) {
            function = self.functions.items.len;
            try self.function_ops.append(self.allocator, op);
            const selected = for (self.selected) |candidate| {
                if (candidate.ptr == op.ptr) break true;
            } else false;
            const init = attr(op, "ora.init");
            if (!mlir.oraAttributeIsNull(init) and !mlir.mlirAttributeIsABool(init)) return error.InvalidLifecycleConstructor;
            try self.functions.append(self.allocator, .{
                .owner = (try self.string(op, "ora.source_owner_key")) orelse return error.InvalidLifecycleOwner,
                .module_path = (try self.string(op, "ora.source_module_path")) orelse return error.InvalidLifecycleOwner,
                .symbol = (try self.string(op, "sym_name")) orelse return error.InvalidLifecycleOwner,
                .ordinal = ordinal,
                .runtime = selected,
                .constructor = !mlir.oraAttributeIsNull(init) and mlir.mlirAttributeIsABool(init) and mlir.mlirBoolAttrGetValue(init),
                .hir_sha256 = try operationHash(op),
                .events = &.{},
            });
        } else if (function) |index| {
            const effect = mlir.oraOperationGetMemoryEffectKind(op);
            const call = std.mem.eql(u8, name, "func.call") or std.mem.eql(u8, name, "func.call_indirect") or std.mem.eql(u8, name, "ora.external_call");
            if (call or effect == mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE or effect == mlir.ORA_OPERATION_MEMORY_EFFECT_UNKNOWN) {
                var target = try self.string(op, "global");
                const callee = attr(op, "callee");
                if (call and !mlir.oraAttributeIsNull(callee) and mlir.mlirAttributeIsAFlatSymbolRef(callee)) {
                    const text = mlir.mlirFlatSymbolRefAttrGetValue(callee);
                    target = try self.allocator.dupe(u8, text.data[0..text.length]);
                }
                const event: data.Event = .{
                    .ordinal = ordinal,
                    .operation = try self.allocator.dupe(u8, name),
                    .kind = if (call) .call else if (effect == mlir.ORA_OPERATION_MEMORY_EFFECT_WRITE) .write else .unknown,
                    .target = target,
                    .refusal = if (std.mem.eql(u8, name, "ora.external_call")) .external_interference else if (!call and effect == mlir.ORA_OPERATION_MEMORY_EFFECT_UNKNOWN) .unknown_effect else null,
                    .hir_sha256 = try operationHash(op),
                };
                self.functions.items[index].events = try std.mem.concat(self.allocator, data.Event, &.{ self.functions.items[index].events, &.{event} });
            }
        }
        for (0..@intCast(mlir.oraOperationGetNumRegions(op))) |region_index| {
            const region = mlir.oraOperationGetRegion(op, region_index);
            var block = mlir.oraRegionGetFirstBlock(region);
            while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
                var child = mlir.oraBlockGetFirstOperation(block);
                while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) try self.walk(child, owner, function);
            }
        }
    }
};

fn attr(op: mlir.MlirOperation, name: []const u8) mlir.MlirAttribute {
    return mlir.oraOperationGetAttributeByName(op, mlir.oraStringRefCreate(name.ptr, name.len));
}

fn limbs(parts: [4]u64) u256 {
    var value: u256 = 0;
    for (parts, 0..) |part, index| value |= @as(u256, part) << @as(u8, @intCast(index * 64));
    return value;
}

fn integer(value: mlir.MlirAttribute) !u256 {
    const text = mlir.oraIntegerAttrGetValueString(value);
    defer if (text.data != null) mlir.oraStringRefFree(text);
    if (text.data == null) return error.InvalidLifecycleInteger;
    return std.fmt.parseInt(u256, text.data[0..text.length], 10) catch error.InvalidLifecycleInteger;
}

fn operationHash(op: mlir.MlirOperation) !data.Digest {
    const printed = mlir.oraOperationPrintToString(op);
    defer if (printed.data != null) mlir.oraStringRefFree(printed);
    if (printed.data == null or printed.length == 0) return error.MissingLifecycleBody;
    return data.hash(printed.data[0..printed.length]);
}
