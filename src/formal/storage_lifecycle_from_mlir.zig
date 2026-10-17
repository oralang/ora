//! Fresh, non-authorizing inventory. Submitted rows are compared to live HIR;
//! effects and bodies are recorded, not declared semantically verified.
const std = @import("std");
const mlir = @import("mlir_c_api").c;
const z3 = @import("ora_z3_verification");
const builtin = @import("ora_types").builtin;
const data = @import("storage_lifecycle.zig");
const obligation = @import("obligation.zig");
const canonical = @import("obligation_from_mlir.zig");
const lean = @import("obligation_to_lean.zig");

pub const Snapshot = struct {
    arena: std.heap.ArenaAllocator,
    inventory: data.Inventory,
    function_ops: []const mlir.MlirOperation,
    operations: []const mlir.MlirOperation,

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
    return .{ .arena = arena, .inventory = inventory, .function_ops = collector.function_ops.items, .operations = collector.operations.items };
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

/// A single direct contract scope with all live roots and effects accounted
/// for. Every selected guard names the actual retained enforcement operation.
pub fn proofInputs(projection: *Projection) !data.ProofInputs {
    const snapshot = &projection.snapshot;
    const arena = snapshot.arena.allocator();
    const inventory = snapshot.inventory;
    try projection.canonical.set.validateIdReferences();
    if (inventory.claims.len == 0) return error.MissingLifecycleClaim;
    const slots = try arena.alloc(data.SlotBinding, inventory.claims.len);
    const scope = mlir.mlirOperationGetParentOperation(snapshot.operations[inventory.claims[0].ordinal]);
    const scope_ordinal = for (snapshot.operations, 0..) |op, i| {
        if (op.ptr == scope.ptr) break i;
    } else return error.InvalidLifecycleOwner;
    for (snapshot.operations) |op| {
        if (mlir.mlirOperationGetParentOperation(op).ptr == scope.ptr and operationIs(op, "ora.invariant"))
            return error.UnauthenticatedLifecycleContractInvariant;
    }
    for (inventory.claims, slots, 0..) |claim, *slot, i| {
        const global = snapshot.operations[claim.ordinal];
        if (mlir.mlirOperationGetParentOperation(global).ptr != scope.ptr) return error.AmbiguousLifecycleOwner;
        slot.* = .{ .claim_index = @intCast(i), .global_ordinal = claim.ordinal, .owner_ordinal = @intCast(scope_ordinal) };
    }
    var guards: std.ArrayList(data.GuardBinding) = .empty;
    const rows = projection.canonical.set.function_summaries;
    if (projection.bindings.len != rows.len) return error.MissingLifecycleBinding;
    for (projection.bindings, rows) |binding, row| {
        const function = for (inventory.functions) |function| {
            if (function.ordinal == binding.function_ordinal) break function;
        } else return error.ForeignLifecycleBinding;
        const op = snapshot.operations[function.ordinal];
        if (function.constructor or mlir.mlirOperationGetParentOperation(op).ptr != scope.ptr)
            return error.UnsupportedLifecycleRoot;
        if (row.id != binding.function_id or row.projection_completeness != .complete or !row.write_footprint_complete or row.paths.len != 1)
            return error.UnsupportedLifecycleRoot;
        _ = try authenticateEntry(projection, row, function);
        if (!std.mem.eql(u8, &(try lean.functionVcSha256(projection.canonical.set, row.id)), &binding.function_vc_sha256))
            return error.LifecycleDigestMismatch;
        var writer_events: std.ArrayList(data.Event) = .empty;
        for (function.events) |effect| {
            if (std.mem.eql(u8, effect.operation, "ora.sstore") and effect.kind == .write and effect.refusal == null) {
                try writer_events.append(arena, effect);
                continue;
            }
            // These control operations lack MemoryEffectOpInterface. Account
            // for their exact canonical control role; never discard an unknown effect.
            const live = snapshot.operations[effect.ordinal];
            if (mlir.mlirOperationGetParentOperation(live).ptr != op.ptr or mlir.oraOperationGetNumRegions(live) != 0)
                return error.UnsupportedLifecycleEffect;
            if (std.mem.eql(u8, effect.operation, "ora.return") and row.paths[0].returns == .direct and
                mlir.oraOperationIsNull(mlir.oraOperationGetNextInBlock(live))) continue;
            // authenticateEntry already matched every semantic requires to
            // its adjacent retained runtime assertion by live identity.
            if (std.mem.eql(u8, effect.operation, "ora.requires") and operationIs(live, "ora.requires")) continue;
            if (!std.mem.eql(u8, effect.operation, "ora.refinement_guard") and !std.mem.eql(u8, effect.operation, "ora.assert"))
                return error.UnsupportedLifecycleEffect;
            const key = try std.fmt.allocPrint(arena, "op:{d}/{s}", .{ effect.ordinal, effect.operation });
            for (row.paths[0].timeline) |event| {
                if (event.payload == .safety_checkpoint and std.mem.eql(u8, event.structural_key, key)) break;
            } else return error.LifecycleGuardMismatch;
        }
        var writes: usize = 0;
        for (row.paths[0].timeline, 0..) |event, index| switch (event.payload) {
            .storage_read, .storage_write => {
                const location = if (event.payload == .storage_read) event.payload.storage_read.location else event.payload.storage_write.location;
                if (location.projections.len != 0) return error.UnsupportedLifecycleLocation;
                if (event.payload == .storage_write) {
                    if (writes >= writer_events.items.len) return error.MissingLifecycleWriter;
                    const effect = writer_events.items[writes];
                    const key = try std.fmt.allocPrint(arena, "op:{d}/{s}", .{ effect.ordinal, effect.operation });
                    if (effect.kind != .write or effect.refusal != null or !std.mem.eql(u8, effect.operation, "ora.sstore") or
                        !std.mem.eql(u8, effect.target orelse "", location.root) or !std.mem.eql(u8, key, event.structural_key))
                        return error.LifecycleWriterMismatch;
                    writes += 1;
                }
            },
            .safety_checkpoint => |checkpoint| {
                if (event.id != index) return error.LifecycleGuardMismatch;
                if (checkpoint.formulas.len != 1) return error.UnsupportedLifecycleGuard;
                for (function.events) |effect| {
                    if (!std.mem.eql(u8, effect.operation, "ora.refinement_guard")) continue;
                    const key = try std.fmt.allocPrint(arena, "op:{d}/{s}", .{ effect.ordinal, effect.operation });
                    if (!std.mem.eql(u8, key, event.structural_key)) continue;
                    const live = snapshot.operations[effect.ordinal];
                    if (mlir.mlirOperationGetParentOperation(live).ptr != op.ptr)
                        return error.LifecycleGuardMismatch;
                    try guards.append(arena, .{ .function_id = row.id, .event_id = event.id, .operation_ordinal = effect.ordinal, .hir_sha256 = try operationHash(live) });
                    break;
                }
            },
            else => return error.UnsupportedLifecycleEvent,
        };
        if (writes != writer_events.items.len) return error.MissingLifecycleWriter;
    }
    const inputs: data.ProofInputs = .{ .inventory = inventory, .functions = projection.bindings, .slots = slots, .guards = guards.items };
    _ = try inputs.sha256(arena);
    return inputs;
}

/// Exact live enforcement, not a source tag or an accepted accounting row.
/// Returns the end of the unchanged-store prefix containing every mirror.
pub fn authenticateEntry(projection: *Projection, row: obligation.FunctionSummaryRow, function: data.Function) !usize {
    if (row.entry_invariant_ids.len != 0) return error.UnauthenticatedLifecycleContractInvariant;
    try projection.canonical.set.validateIdReferences();
    const operations = projection.snapshot.operations;
    if (row.paths.len != 1 or function.ordinal >= operations.len) return error.UnsupportedLifecycleRoot;
    const owner = operations[function.ordinal];
    const parent = mlir.mlirOperationGetParentOperation(owner);
    const scope = for (operations, 0..) |op, index| {
        if (op.ptr == parent.ptr and operationIs(op, "ora.contract")) break index;
    } else return error.InvalidLifecycleOwner;
    const arena = projection.snapshot.arena.allocator();
    if (!std.mem.eql(u8, row.entry_scope orelse "", try std.fmt.allocPrint(arena, "op:{d}/ora.contract", .{scope})) or
        !std.mem.eql(u8, row.structural_key, try std.fmt.allocPrint(arena, "op:{d}/func.func", .{function.ordinal})) or
        row.owner != .function or !std.mem.eql(u8, row.owner.function.name, function.symbol)) return error.InvalidLifecycleOwner;
    const bindings = projection.canonical.source_fact_bindings;
    var end: usize = 0;
    var previous: u32 = 0;
    for (row.assumption_ids) |id| {
        var selected: ?canonical.SourceFactOpBinding = null;
        for (bindings) |binding| {
            if (std.mem.indexOfScalar(obligation.Id, binding.assumption_ids, id) == null) continue;
            if (selected != null) return error.UnauthenticatedLifecycleRequires;
            selected = binding;
        }
        const source = selected orelse return error.UnauthenticatedLifecycleRequires;
        const fact = obligation.findById(projection.canonical.set.assumptions, id) orelse return error.UnauthenticatedLifecycleRequires;
        if (fact.origin != .mlir_op or fact.origin.mlir_op.ordinal != source.op_ordinal or
            !std.mem.eql(u8, fact.origin.mlir_op.op_name, "ora.requires")) return error.UnauthenticatedLifecycleRequires;
        if (source.assumption_ids.len != 1 or source.op_ordinal >= operations.len or source.op_ordinal <= previous or
            !entryBindingOwned(source, function, "assumption_context")) return error.UnauthenticatedLifecycleRequires;
        const semantic = operations[source.op_ordinal];
        const mirror = mlir.oraOperationGetNextInBlock(semantic);
        if (@intFromPtr(semantic.ptr) != source.source_op_id or !operationIs(semantic, "ora.requires") or
            mlir.mlirOperationGetParentOperation(semantic).ptr != owner.ptr or mlir.oraOperationIsNull(mirror) or
            !operationIs(mirror, "ora.assert") or mlir.mlirOperationGetParentOperation(mirror).ptr != owner.ptr or
            mlir.oraOperationGetNumOperands(semantic) != 1 or mlir.oraOperationGetNumOperands(mirror) != 1 or
            mlir.oraOperationGetOperand(semantic, 0).ptr != mlir.oraOperationGetOperand(mirror, 0).ptr)
            return error.UnauthenticatedLifecycleRequires;
        var runtime: ?canonical.SourceFactOpBinding = null;
        for (bindings) |binding| {
            if (binding.source_fact_id != source.source_fact_id or !entryBindingOwned(binding, function, "runtime_condition")) continue;
            if (runtime != null) return error.UnauthenticatedLifecycleRequires;
            runtime = binding;
        }
        const checked = runtime orelse return error.UnauthenticatedLifecycleRequires;
        if (checked.op_ordinal != source.op_ordinal + 1 or checked.op_ordinal >= operations.len or operations[checked.op_ordinal].ptr != mirror.ptr or
            checked.source_op_id != @intFromPtr(mirror.ptr) or !checked.runtime_check_present or
            checked.runtime_check_ids.len != 1) return error.UnauthenticatedLifecycleRequires;
        var producers: usize = 0;
        for (projection.canonical.runtime_check_producers) |producer| {
            if (producer.id != checked.runtime_check_ids[0]) continue;
            if (producer.source_op_id != checked.source_op_id or producer.op_ordinal != checked.op_ordinal)
                return error.UnauthenticatedLifecycleRequires;
            producers += 1;
        }
        if (producers != 1) return error.UnauthenticatedLifecycleRequires;
        const key = try std.fmt.allocPrint(projection.snapshot.arena.allocator(), "op:{d}/ora.assert", .{checked.op_ordinal});
        var matches: usize = 0;
        for (row.paths[0].timeline, 0..) |event, index| {
            if (!std.mem.eql(u8, event.structural_key, key)) continue;
            if (index < end or event.id != index or event.payload != .safety_checkpoint or
                event.payload.safety_checkpoint.formulas.len != 1) return error.UnauthenticatedLifecycleRequires;
            end = index + 1;
            matches += 1;
        }
        if (matches != 1) return error.UnauthenticatedLifecycleRequires;
        previous = checked.op_ordinal;
    }
    var actual: usize = 0;
    for (operations) |op| {
        if (mlir.mlirOperationGetParentOperation(op).ptr == owner.ptr and operationIs(op, "ora.requires")) actual += 1;
    }
    if (actual != row.assumption_ids.len) return error.UnauthenticatedLifecycleRequires;
    for (row.paths[0].timeline[0..end]) |event| switch (event.payload) {
        .storage_read, .safety_checkpoint => {},
        else => return error.LifecycleEntryStoreChanged,
    };
    return end;
}

fn entryBindingOwned(binding: canonical.SourceFactOpBinding, function: data.Function, role: []const u8) bool {
    return std.mem.eql(u8, binding.kind, "requires") and binding.roles.len == 1 and std.mem.eql(u8, binding.roles[0], role) and
        std.mem.eql(u8, binding.module_path orelse "", function.module_path) and
        std.mem.eql(u8, binding.owner_key orelse "", function.owner) and std.mem.eql(u8, binding.runtime_symbol orelse "", function.symbol);
}

fn operationIs(op: mlir.MlirOperation, expected: []const u8) bool {
    const name = mlir.oraOperationGetName(op);
    defer if (name.data != null) mlir.oraStringRefFree(name);
    return name.data != null and std.mem.eql(u8, name.data[0..name.length], expected);
}

const Collector = struct {
    allocator: std.mem.Allocator,
    selected: []const mlir.MlirOperation,
    claims: std.ArrayList(data.Claim) = .empty,
    functions: std.ArrayList(data.Function) = .empty,
    function_ops: std.ArrayList(mlir.MlirOperation) = .empty,
    operations: std.ArrayList(mlir.MlirOperation) = .empty,
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
        try self.operations.append(self.allocator, op);
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
