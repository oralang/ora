//! Lifecycle data and bounded proof terms. Digests and unchecked terms grant no authority.
const std = @import("std");
const builtin = @import("ora_types").builtin;
const data = @import("storage_lifecycle.zig");
const obligation = @import("obligation.zig");
const lean = @import("obligation_to_lean.zig");

pub fn writeProjection(writer: anytype, allocator: std.mem.Allocator, inventory: data.Inventory, set: obligation.ObligationSet, bindings: []const data.FunctionBinding) !void {
    try writeProjectionNamed(writer, allocator, inventory, set, bindings, "Ora.Generated.LifecycleFunctions");
}

fn writeProjectionNamed(writer: anytype, allocator: std.mem.Allocator, inventory: data.Inventory, set: obligation.ObligationSet, bindings: []const data.FunctionBinding, namespace: []const u8) !void {
    try data.validateBindings(inventory, bindings);
    if (bindings.len != set.function_summaries.len) return error.MissingLifecycleBinding;
    for (bindings) |binding| {
        const row = obligation.findById(set.function_summaries, binding.function_id) orelse return error.ForeignLifecycleBinding;
        const function = for (inventory.functions) |candidate| {
            if (candidate.ordinal == binding.function_ordinal) break candidate;
        } else return error.ForeignLifecycleBinding;
        if (row.owner != .function or !std.mem.eql(u8, row.owner.function.name, function.symbol)) return error.ForeignLifecycleBinding;
        if (!std.mem.eql(u8, &(try lean.functionVcSha256(set, row.id)), &binding.function_vc_sha256)) return error.LifecycleDigestMismatch;
    }
    try lean.writeCallCustodyModule(writer, set, namespace);
    try writer.print("\nnamespace {s}\ndef lifecycleInventorySha256 : String := \"{s}\"\ndef lifecycleBindings : List (Nat × Nat × String) := [", .{ namespace, std.fmt.bytesToHex(try inventory.sha256(allocator), .lower) });
    for (bindings, 0..) |binding, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("({d}, {d}, \"{s}\")", .{ binding.function_ordinal, binding.function_id, std.fmt.bytesToHex(binding.function_vc_sha256, .lower) });
    }
    try writer.print("]\nend {s}\n", .{namespace});
}

pub fn writeClaim(writer: anytype, allocator: std.mem.Allocator, inventory: data.Inventory, index: usize) !void {
    try inventory.validate();
    if (index >= inventory.claims.len) return error.MissingLifecycleClaim;
    const claim = inventory.claims[index];
    if (claim.refusal != null) return error.UnsupportedLifecycleClaim;
    const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
    const bytes = try inventory.canonical(allocator);
    defer allocator.free(bytes);
    const digest = try inventory.sha256(allocator);
    try writer.writeAll("import Ora.Storage.Lifecycle\nnamespace Ora.Generated.LifecycleInputs\nopen Ora.Obligation Ora.Loop\n");
    try writer.print("def inventorySha256 : String := \"{s}\"\n", .{std.fmt.bytesToHex(digest, .lower)});
    // Hex preserves every canonical byte, including paths and unsupported rows.
    try writer.writeAll("def inventoryBytesHex : String := \"");
    for (bytes) |byte| try writer.print("{x:0>2}", .{byte});
    try writer.writeAll("\"\n");
    try writer.print("def claimIndex : Nat := {d}\ndef candidateClaim : Ora.StorageLifecycle.Claim := {{\n  location := {{ root := ", .{index});
    const root = try std.fmt.allocPrint(allocator, "{s}/slot:{s}", .{ claim.owner, claim.root });
    defer allocator.free(root);
    try string(writer, root);
    try writer.writeAll(" }\n  name := ");
    try string(writer, @tagName(claim.bounds.?.kind));
    try writer.writeAll("\n  bounds := [");
    const bounds = claim.bounds.?;
    if (bounds.kind != .MaxValue) try value(writer, info, bounds.minimum);
    if (bounds.kind == .InRange) try writer.writeAll(", ");
    if (bounds.kind != .MinValue) try value(writer, info, bounds.maximum);
    try writer.writeAll("]\n}\ndef zeroAtClaim : Value := ");
    try value(writer, info, 0);
    try writer.writeAll("\nend Ora.Generated.LifecycleInputs\n");
}

fn value(writer: anytype, info: builtin.BuiltinIntegerInfo, literal: u256) !void {
    try writer.print(".integer (.ofNat (.{s} .w{d}) {d})", .{ if (info.signed) "signed" else "unsigned", info.width, literal });
}

fn string(writer: anytype, text: []const u8) !void {
    try writer.writeByte('"');
    for (text) |byte| switch (byte) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0...8, 11, 12, 14...31, 127 => return error.InvalidLifecycleString,
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

/// Targets are regenerated separately from proof terms. The equations bind
/// the successful view to the unmodified canonical timeline at the kernel.
pub fn writeProducerTargets(writer: anytype, allocator: std.mem.Allocator, inputs: data.ProofInputs, set: obligation.ObligationSet) !void {
    try writer.writeAll("import Ora.Storage.CallerIdentity\n");
    try writeProjectionNamed(writer, allocator, inputs.inventory, set, inputs.functions, "Ora.Generated.Obligations.Lifecycle");
    try writer.print("\nnamespace Ora.Generated.Obligations.Lifecycle\nopen Ora.Obligation Ora.Loop Ora.StorageLifecycle Ora.CallerIdentityModel\ndef lifecycleInputsSha256 : String := \"{s}\"\n", .{std.fmt.bytesToHex(try inputs.sha256(allocator), .lower)});
    for (inputs.inventory.claims, 0..) |claim, ci| {
        const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
        try writer.print("namespace LC{d}\ndef word (bits : BitVec {d}) : Value := .integer {{shape := .unsigned .w{d}, bits}}\ndef claim : Claim := {{location := {{root := ", .{ ci, info.width, info.width });
        try string(writer, claim.root);
        try writer.print("}}, name := \"{s}\", bounds := [", .{@tagName(claim.bounds.?.kind)});
        if (claim.bounds.?.kind != .MaxValue) try value(writer, info, claim.bounds.?.minimum);
        if (claim.bounds.?.kind == .InRange) try writer.writeAll(", ");
        if (claim.bounds.?.kind != .MinValue) try value(writer, info, claim.bounds.?.maximum);
        try writer.writeAll("]}\n");
        for (set.function_summaries) |row| {
            _ = try writerRecipe(claim, row, inputs);
            try writer.print("namespace F{d}\nabbrev row := emittedFunctionSummary_{d}\ndef original : List AdapterEvent := [", .{ row.id, row.id });
            try producerEvents(writer, row, inputs, false);
            try writer.writeAll("]\ndef selected : List Nat := [");
            var first = true;
            for (inputs.guards) |guard| if (guard.function_id == row.id) {
                if (!first) try writer.writeAll(", ");
                first = false;
                try writer.print("{d}", .{guard.event_id});
            };
            try writer.writeAll("]\ndef events : List AdapterEvent := [");
            try producerEvents(writer, row, inputs, true);
            try writer.writeAll("]\ndef strictEvents : List AdapterEvent := original.map fun event => match event with | .check f => .assume f | other => other\n");
            try writer.print("def entryPrefix : List AdapterEvent := original.take {d}\n", .{try entryPrefixEnd(allocator, set, row)});
            try writer.print("def invocationKey (serial : Nat) : InvocationKey := {{moduleDigest := \"{s}\", functionId := {d}, serial}}\n", .{ std.fmt.bytesToHex(inputs.inventory.module_sha256, .lower), row.id });
            try writer.writeAll(
                \\def callerEnv (capture : Capture) : Env := (capturedEnv capture).toEnv (invocationKey capture.key.serial)
                \\def semantics (capture : Capture) := (adapterSemantics {} (directStorageContext emittedManifest (callerEnv capture) [])).withCallChecks
                \\def prefixRoot : Family (CallFormula AdapterFormula FormulaRef AdapterChild) StorageLocationExpr FormulaRef AdapterChild := {semantics, events := entryPrefix, entry := fun capture state =>
                \\  (capture.key = invocationKey capture.key.serial ∧ ∃ input, row.callSignature.entry? input = some state)}
                \\def root : Family (CallFormula AdapterFormula FormulaRef AdapterChild) StorageLocationExpr FormulaRef AdapterChild := {semantics, events, entry := fun capture state =>
                \\  (capture.key = invocationKey capture.key.serial ∧ ∃ input, row.callSignature.entry? input = some state ∧
                \\    row.entryFactsHold emittedManifest input (callerEnv capture).invocationCaller)}
                \\def strictWitness : Prop := ∃ capture input state result,
                \\  capture.key = invocationKey capture.key.serial ∧ row.callSignature.entry? input = some state ∧
                \\  row.entryFactsHold emittedManifest input (callerEnv capture).invocationCaller ∧ Invariant [claim] state.loop.storage ∧
                \\  TimelineExec (semantics capture) strictEvents state result
                \\def target : Prop := row.paths.length = 1 ∧
                \\  emittedCallManifest.timeline {} (row.paths[0]'(by decide)).timeline = .ok original ∧
                \\  successfulView original selected = some events ∧ prefixRoot.Checked [claim] ∧
                \\  root.Checked [claim] ∧ claim.holds (word 0) ∧ strictWitness
                \\
            );
            try writer.print("end F{d}\n", .{row.id});
        }
        try writer.writeAll("def roots := [");
        for (set.function_summaries, 0..) |row, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("F{d}.prefixRoot, F{d}.root", .{ row.id, row.id });
        }
        try writer.writeAll("]\ndef initial (store : StorageState) : Prop := store claim.location = word 0\ndef certificateTarget : Prop := Certificate [claim] initial (roots.map Family.step)\n");
        try writer.print("end LC{d}\n", .{ci});
    }
    try writer.writeAll("def emittedLifecycle : Prop := ");
    for (inputs.inventory.claims, 0..) |_, ci| {
        for (set.function_summaries) |row| try writer.print("LC{d}.F{d}.target ∧ ", .{ ci, row.id });
        try writer.print("LC{d}.certificateTarget ∧ ", .{ci});
    }
    try writer.writeAll("True\nend Ora.Generated.Obligations.Lifecycle\n");
}

// The live validator authenticates the adjacent requires/assert pair. Bind
// the target's failed-entry prefix to that exact canonical operation ordinal.
fn entryPrefixEnd(allocator: std.mem.Allocator, set: obligation.ObligationSet, row: obligation.FunctionSummaryRow) !usize {
    if (row.entry_invariant_ids.len != 0) return error.UnauthenticatedLifecycleContractInvariant;
    var end: usize = 0;
    for (row.assumption_ids) |id| {
        const fact = obligation.findById(set.assumptions, id) orelse return error.UnauthenticatedLifecycleRequires;
        if (fact.origin != .mlir_op or !std.mem.eql(u8, fact.origin.mlir_op.op_name, "ora.requires")) return error.UnauthenticatedLifecycleRequires;
        const key = try std.fmt.allocPrint(allocator, "op:{d}/ora.assert", .{fact.origin.mlir_op.ordinal + 1});
        defer allocator.free(key);
        var matches: usize = 0;
        for (row.paths[0].timeline, 0..) |event, index| {
            if (!std.mem.eql(u8, event.structural_key, key)) continue;
            if (index < end or event.payload != .safety_checkpoint) return error.UnauthenticatedLifecycleRequires;
            end = index + 1;
            matches += 1;
        }
        if (matches != 1) return error.UnauthenticatedLifecycleRequires;
    }
    for (row.paths[0].timeline[0..end]) |event| switch (event.payload) {
        .safety_checkpoint, .storage_read => {},
        else => return error.LifecycleEntryStoreChanged,
    };
    return end;
}

fn writerRecipe(claim: data.Claim, row: obligation.FunctionSummaryRow, inputs: data.ProofInputs) !?usize {
    if (row.paths.len != 1) return error.UnsupportedLifecycleRoot;
    var written_at: ?usize = null;
    for (row.paths[0].timeline, 0..) |event, index| {
        if (event.payload != .storage_write) continue;
        const write = event.payload.storage_write;
        if (!std.mem.eql(u8, write.location.root, claim.root)) continue;
        if (written_at != null) return error.UnsupportedLifecycleWriter;
        written_at = index;
        if (index == 0 or row.parameters.len != 1 or !obligation.typeRefEqual(row.parameters[0].ty, .{ .compiler_type_id = claim.type_id.? }))
            return error.UnsupportedLifecycleWriter;
        for (inputs.guards) |guard| {
            if (guard.function_id == row.id and guard.event_id == 0) break;
        } else return error.MissingLifecycleRetainedGuard;
    }
    if (written_at) |index| for (row.paths[0].timeline[1..index]) |event| {
        if (event.payload != .safety_checkpoint) return error.UnsupportedLifecycleWriter;
    };
    return written_at;
}

fn producerEvents(writer: anytype, row: obligation.FunctionSummaryRow, inputs: data.ProofInputs, successful: bool) !void {
    for (row.paths[0].timeline, 0..) |event, index| {
        if (index != 0) try writer.writeAll(", ");
        switch (event.payload) {
            .safety_checkpoint => |checkpoint| {
                const selected = for (inputs.guards) |guard| {
                    if (guard.function_id == row.id and guard.event_id == event.id) break true;
                } else false;
                if (checkpoint.formulas.len != 1 or checkpoint.formulas[0].formula != .term) return error.UnsupportedLifecycleGuard;
                try writer.print(".{s} (.ordinary (.inl (.term {d})))", .{ if (successful and selected) "assume" else "check", checkpoint.formulas[0].formula.term });
            },
            .storage_read, .storage_write => {
                const read = event.payload == .storage_read;
                const location = if (read) event.payload.storage_read.location else event.payload.storage_write.location;
                if (location.projections.len != 0 or location.value_ty != .compiler_type_id) return error.UnsupportedLifecycleLocation;
                try writer.print(".storage (.{s} ", .{if (read) "read" else "write"});
                if (read) try writer.print("{{file_id := {d}, pattern_id := {d}}} ", .{ event.payload.storage_read.result.file_id, event.payload.storage_read.result.pattern_id });
                try writer.writeAll("{root := ");
                try string(writer, location.root);
                try writer.print(", projections := [], valueTy := .compilerTypeId {d}}}", .{location.value_ty.compiler_type_id});
                if (!read) {
                    const formula = event.payload.storage_write.value.formula;
                    if (formula != .term) return error.UnsupportedLifecycleWriter;
                    try writer.print(" (.term {d})", .{formula.term});
                }
                try writer.writeByte(')');
            },
            else => return error.UnsupportedLifecycleEvent,
        }
    }
}

/// Bounded structural proofs over fresh data, independent of any solver result.
/// The separate target module is the type checked by custody.
pub fn writeProducerProofs(writer: anytype, allocator: std.mem.Allocator, inputs: data.ProofInputs, set: obligation.ObligationSet) !void {
    _ = try inputs.sha256(allocator);
    try writer.writeAll("\nnamespace Ora.Generated.Obligations.Lifecycle\nopen Ora.Obligation Ora.Loop Ora.StorageLifecycle Ora.CallerIdentityModel\n");
    for (inputs.inventory.claims, 0..) |claim, ci| {
        const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
        try writer.print("namespace LC{d}\ntheorem zero_valid : claim.holds (word 0) := by\n  change ", .{ci});
        if (claim.bounds.?.kind == .InRange) try writer.writeAll("(0 : Nat) ≤ 0 ∧ ");
        if (claim.bounds.?.kind == .MinValue) try writer.writeAll("(0 : Nat) ≤ 0") else try writer.print("(0 : Nat) ≤ {d}", .{claim.bounds.?.maximum});
        try writer.writeAll("\n  decide\n");
        for (set.function_summaries) |row| {
            try writer.print("namespace F{d}\n", .{row.id});
            try writer.writeAll("theorem entry_selected : row.entryFormulas? emittedManifest = some [");
            for (row.assumption_ids, 0..) |id, index| {
                if (index != 0) try writer.writeAll(", ");
                const formula = (obligation.findById(set.assumptions, id) orelse return error.UnauthenticatedLifecycleRequires).formula.?;
                if (formula != .term) return error.UnsupportedLifecycleFormula;
                try writer.print(".term {d}", .{formula.term});
            }
            try writer.writeAll("] := by unfold emittedManifest; rfl\n");
            try writer.writeAll("attribute [local irreducible] emittedManifest\n");
            const guarded = try writerRecipe(claim, row, inputs);
            if (guarded) |write_index| {
                const parameter = row.parameters[0];
                const identity = parameter.id orelse return error.UnsupportedLifecycleParameter;
                const checkpoint = row.paths[0].timeline[0].payload.safety_checkpoint.formulas[0].formula.term;
                const written = row.paths[0].timeline[write_index].payload.storage_write.value.formula.term;
                try writer.print("def entryState (bits : BitVec {d}) (store : StorageState) : StorageActionState := {{loop := {{locals := [], storage := store}}, entryStorage := store, reads := [{{id := {{file_id := {d}, pattern_id := {d}}}, value := word bits}}]}}\n", .{ info.width, identity.file_id, identity.pattern_id });
                try writer.print("theorem entry_shape (input : CallInput) (state : StorageActionState)\n    (entry : row.callSignature.entry? input = some state) :\n    ∃ bits : BitVec {d}, state = entryState bits input.storage := by\n", .{info.width});
                try writer.writeAll(
                    \\  rcases input with ⟨args, store⟩
                    \\  have typed : row.callSignature.inputTyped {arguments := args, storage := store} = true := by
                    \\    unfold CallSignature.entry? at entry
                    \\    split at entry <;> simp_all
                    \\
                );
                try writer.print("  change (decide ([] : List FreeVarId).Nodup && (1 == args.length &&\n    ([(TyRef.compilerTypeId {d})].zip args).all (fun (ty, v) => storageValueMatchesTy v ty))) = true at typed\n", .{claim.type_id.?});
                try writer.writeAll(
                    \\  cases args with
                    \\  | nil => simp at typed
                    \\  | cons value tail =>
                    \\    have tailNil : tail = [] := by
                    \\      cases tail with
                    \\      | nil => rfl
                    \\      | cons _ _ => simp at typed
                    \\    subst tail
                    \\    cases value with
                    \\    | bool _ => simp [storageValueMatchesTy, TyRef.isBool, compilerTypeIdBool, Ora.Spec.expectedCompilerTypeIdBool] at typed
                    \\    | integer value =>
                    \\      rcases value with ⟨shape, bits⟩
                    \\
                );
                try writer.print("      have same : shape = .unsigned .w{d} := by symm; simpa [storageValueMatchesTy] using typed\n      subst shape\n      exact ⟨bits, (Option.some.inj entry).symm⟩\n", .{info.width});
                try writer.writeAll("theorem preserved : root.Checked [claim] := by\n  intro capture state ⟨_, input, entry, _⟩ valid\n  obtain ⟨bits, rfl⟩ := entry_shape input state entry\n");
                try writer.print("  apply guarded_write_after_checks (semantics capture) (.ordinary (.inl (.term {d}))) [", .{checkpoint});
                for (row.paths[0].timeline[1..write_index], 0..) |event, index| {
                    if (index != 0) try writer.writeAll(", ");
                    try writer.print(".ordinary (.inl (.term {d}))", .{event.payload.safety_checkpoint.formulas[0].formula.term});
                }
                try writer.writeAll("] valid (value := word bits)\n    (after := {state := {entryState bits input.storage with loop := {locals := [], storage := input.storage.set claim.location (word bits)}}, writeTrace := [claim.location]})\n");
                try writer.writeAll("  ·\n");
                try producerFormulaBridge(writer, set, checkpoint, "    ", "capture");
                try writer.writeAll("  · ");
                try writer.print("change execStorageAction? (semantics capture).storage (.write {{root := ", .{});
                try string(writer, claim.root);
                try writer.print(", projections := [], valueTy := .compilerTypeId {d}}} (.term {d})) _ = _\n", .{ claim.type_id.?, written });
                try producerWriteBridge(writer, set, written, "    ", "capture");
                try writer.writeAll("  · rfl\n  ·\n");
                try producerFrames(writer, row.paths[0].timeline[write_index + 1 ..], "    ");
            } else {
                try writer.writeAll(
                    \\theorem preserved : root.Checked [claim] := by
                    \\  intro capture state _ valid
                    \\  apply framed_prefixes _ valid
                    \\  intro observed member
                    \\  have equal : observed = claim := by simpa using member
                    \\  subst observed
                    \\
                );
                try producerFrames(writer, row.paths[0].timeline, "  ");
            }
            try writer.writeAll("theorem prefix_preserved : prefixRoot.Checked [claim] := by\n  intro capture state _ valid\n  apply framed_prefixes _ valid\n  intro observed member\n  have equal : observed = claim := by simpa using member\n  subst observed\n  change ∀ event ∈ entryPrefix, _\n");
            try producerFrames(writer, row.paths[0].timeline[0..try entryPrefixEnd(allocator, set, row)], "  ");
            try writer.print("def witnessState (candidate : Nat) : StorageActionState := {{loop := {{locals := [], storage := fun _ => word (BitVec.ofNat {d} candidate)}}, entryStorage := fun _ => word (BitVec.ofNat {d} candidate), reads := [", .{ info.width, info.width });
            for (row.parameters, 0..) |parameter, index| {
                if (index != 0) try writer.writeAll(", ");
                const identity = parameter.id orelse return error.UnsupportedLifecycleParameter;
                try writer.print("{{id := {{file_id := {d}, pattern_id := {d}}}, value := ", .{ identity.file_id, identity.pattern_id });
                try parameterCandidate(writer, parameter.ty);
                try writer.writeByte('}');
            }
            try writer.print("]}}\ndef witnessInput (candidate : Nat) : CallInput := {{storage := fun _ => word (BitVec.ofNat {d} candidate), arguments := [", .{info.width});
            for (row.parameters, 0..) |parameter, index| {
                if (index != 0) try writer.writeAll(", ");
                try parameterCandidate(writer, parameter.ty);
            }
            try writer.writeAll("]}\ndef witnessCapture (candidate : Nat) : Capture := {key := invocationKey 0, caller := BitVec.ofNat 160 candidate}\ntheorem witnessed : strictWitness := by\n  first\n");
            for ([_]u8{ 0, 1 }) |candidate| {
                const capture = try std.fmt.allocPrint(allocator, "(witnessCapture {d})", .{candidate});
                defer allocator.free(capture);
                try writer.print("  | have executed : ∃ result, TimelineExec (semantics (witnessCapture {d})) strictEvents (witnessState {d}) result := by\n      refine ⟨_, ", .{ candidate, candidate });
                for (row.paths[0].timeline) |event| {
                    switch (event.payload) {
                        .safety_checkpoint => |checkpoint| {
                            try writer.writeAll(".cons (.assume (by\n");
                            try producerFormulaBridge(writer, set, checkpoint.formulas[0].formula.term, "          ", capture);
                            try writer.writeAll("        ) (by try simp only [Ora.Integer.unsignedLe, Ora.Integer.unsignedLt]\n              decide)) (");
                        },
                        .storage_write => |write| {
                            try writer.writeAll(".cons (.storage (by\n");
                            try producerWriteBridge(writer, set, write.value.formula.term, "          ", capture);
                            try writer.writeAll("        )) (");
                        },
                        .storage_read => try writer.writeAll(".cons (.storage (by rfl)) ("),
                        else => return error.UnsupportedLifecycleEvent,
                    }
                }
                try writer.writeAll(".nil _");
                for (row.paths[0].timeline) |_| try writer.writeByte(')');
                try writer.writeAll("⟩\n");
                try writer.print("    obtain ⟨result, trace⟩ := executed\n    refine ⟨witnessCapture {d}, witnessInput {d}, witnessState {d}, result, rfl, rfl, ?_, ?_, trace⟩\n    · unfold FunctionSummaryRow.entryFactsHold\n      rw [entry_selected, show row.callSignature.entry? (witnessInput {d}) = some (witnessState {d}) from rfl]\n      intro formula member\n      simp only [List.mem_cons, List.not_mem_nil, or_false] at member\n", .{ candidate, candidate, candidate, candidate, candidate });
                if (row.assumption_ids.len != 0) {
                    try writer.writeAll("      rcases member with ");
                    for (row.assumption_ids, 0..) |_, index| try writer.writeAll(if (index == 0) "rfl" else " | rfl");
                    try writer.writeByte('\n');
                    for (row.assumption_ids) |id| {
                        const term = obligation.findById(set.assumptions, id).?.formula.?.term;
                        try writer.print("      · have proven : ∃ p, (directStorageContext emittedManifest (bindStorageReads (callerEnv (witnessCapture {d})) (witnessState {d}).reads) []).evalFormula? (.term {d}) (witnessState {d}) = some p ∧ p := by\n          unfold emittedManifest\n          exact ⟨_, rfl, by try simp only [Ora.Integer.unsignedLe, Ora.Integer.unsignedLt]; decide⟩\n        obtain ⟨p, evaluated, holds⟩ := proven\n        change ((directStorageContext emittedManifest (bindStorageReads (callerEnv (witnessCapture {d})) (witnessState {d}).reads) []).evalFormula? (.term {d}) (witnessState {d})).getD False\n        rw [evaluated]\n        exact holds\n", .{ candidate, candidate, term, candidate, candidate, candidate, term, candidate });
                    }
                }
                try writer.writeAll("    · intro observed member\n      have equal : observed = claim := by simpa using member\n      subst observed\n      change ");
                if (claim.bounds.?.kind == .InRange) try writer.print("(0 : Nat) ≤ {d} ∧ ", .{candidate});
                if (claim.bounds.?.kind == .MinValue) try writer.print("(0 : Nat) ≤ {d}", .{candidate}) else try writer.print("({d} : Nat) ≤ {d}", .{ candidate, claim.bounds.?.maximum });
                try writer.writeAll("\n      decide\n");
                try writer.writeAll("\n");
            }
            try writer.writeAll("theorem proved : target := ⟨rfl, rfl, rfl, prefix_preserved, preserved, zero_valid, witnessed⟩\n");
            try writer.print("end F{d}\n", .{row.id});
        }
        try writer.writeAll(
            \\theorem certificateProof : certificateTarget := by
            \\  apply caller_family_certificate (claims := [claim]) (initial := initial) roots ⟨fun _ => word 0, rfl⟩
            \\  · intro store starts observed member
            \\    have equal : observed = claim := by simpa using member
            \\    subst observed
            \\    rw [starts]
            \\    exact zero_valid
            \\  · intro observed member
            \\    simp only [roots, List.mem_cons, List.not_mem_nil, or_false] at member
            \\    rcases member with
        );
        try writer.writeByte(' ');
        for (set.function_summaries, 0..) |_, index| try writer.writeAll(if (index == 0) "rfl | rfl" else " | rfl | rfl");
        try writer.writeByte('\n');
        for (set.function_summaries) |row| try writer.print("    · exact F{d}.prefix_preserved\n    · exact F{d}.preserved\n", .{ row.id, row.id });
        try writer.print("end LC{d}\n", .{ci});
    }
    try writer.writeAll("theorem lifecycleProof : emittedLifecycle := by\n  exact ⟨");
    for (inputs.inventory.claims, 0..) |_, ci| {
        for (set.function_summaries) |row| try writer.print("LC{d}.F{d}.proved, ", .{ ci, row.id });
        try writer.print("LC{d}.certificateProof, ", .{ci});
    }
    try writer.writeAll("True.intro⟩\nend Ora.Generated.Obligations.Lifecycle\n");
}

fn parameterCandidate(writer: anytype, ty: obligation.TypeRef) !void {
    if (ty != .compiler_type_id) return error.UnsupportedLifecycleParameter;
    const info = builtin.integerInfoByComptimeTypeId(ty.compiler_type_id) orelse return error.UnsupportedLifecycleParameter;
    try writer.print(".integer (.ofNat (.{s} .w{d}) candidate)", .{ if (info.signed) "signed" else "unsigned", info.width });
}

fn producerFrames(writer: anytype, events: []const obligation.LoopTimelineEvent, indent: []const u8) !void {
    try writer.print("{s}intro event belongs\n", .{indent});
    if (events.len == 0) return writer.print("{s}simp only [root, sourceRoot, prefixRoot, entryPrefix, original, events, List.take, List.not_mem_nil] at belongs\n", .{indent});
    try writer.print("{s}simp only [root, sourceRoot, prefixRoot, entryPrefix, original, events, List.take, List.mem_cons, List.not_mem_nil, or_false] at belongs\n", .{indent});
    try writer.print("{s}rcases belongs with ", .{indent});
    for (events, 0..) |_, index| try writer.writeAll(if (index == 0) "rfl" else " | rfl");
    try writer.writeByte('\n');
    for (events) |event| {
        if (event.payload != .storage_write) {
            try writer.print("{s}· trivial\n", .{indent});
            continue;
        }
        const location = event.payload.storage_write.location;
        try writer.print("{s}· intro state location evaluates\n{s}  have same := direct_location (directStorageContext emittedManifest (callerEnv capture) []) ", .{ indent, indent });
        try string(writer, location.root);
        try writer.print(" (.compilerTypeId {d}) state location evaluates\n{s}  rw [same]\n{s}  decide\n", .{ location.value_ty.compiler_type_id, indent, indent });
    }
}

fn producerFormulaBridge(writer: anytype, set: obligation.ObligationSet, term: obligation.TermId, indent: []const u8, capture: []const u8) !void {
    try writer.print("{s}change (directStorageContext emittedManifest (callerEnv {s}) []).evalFormula? (.term {d}) _ = _\n{s}unfold StorageFormulaContext.evalFormula?\n{s}rw [direct_context_env _ _ _ (by rfl)]\n", .{ indent, capture, term, indent, indent });
    try writer.print("{s}simp only [Option.bind_eq_bind, Option.some_bind, StorageFormulaContext.formulaStorageReady, formulaDenotes?, emitted_terms_length]\n{s}simp only [directStorageContext, emitted_terms_length, StorageFormulaContext.termStorageReady, ", .{ indent, indent });
    try producerLookups(writer, set, term, set.terms.len + 1);
    try writer.print("Bool.and_true, Bool.not_true, Bool.false_eq_true, if_false, Option.pure_def, Option.some_bind]\n{s}simp only [emitted_terms_length, denoteFormula?, denoteValue?, ", .{indent});
    try producerLookups(writer, set, term, set.terms.len + 1);
    try writer.print("Option.some_bind]\n{s}rfl\n", .{indent});
}

fn producerWriteBridge(writer: anytype, set: obligation.ObligationSet, term: obligation.TermId, indent: []const u8, capture: []const u8) !void {
    _ = capture;
    try writer.print("{s}simp only [execStorageAction?, semantics, TimelineSemantics.withCallChecks, adapterSemantics, StorageFormulaContext.actionSemantics, StorageFormulaContext.evalValue?, StorageFormulaContext.stateEnv?, directStorageContext, StorageFormulaContext.formulaStorageReady, formulaValue?, emitted_terms_length, StorageFormulaContext.termStorageReady, denoteValue?, ", .{indent});
    try producerLookups(writer, set, term, set.terms.len + 1);
    try writer.print("Option.pure_def, Option.some_bind, Bool.not_true, Bool.false_eq_true, if_false]\n{s}rfl\n", .{indent});
}

fn producerLookups(writer: anytype, set: obligation.ObligationSet, id: obligation.TermId, fuel: usize) anyerror!void {
    if (fuel == 0 or id >= set.terms.len) return error.InvalidLifecycleFormula;
    try writer.print("emitted_term_{d}_lookup, ", .{id});
    switch (set.terms[id]) {
        .binary => |term| {
            try producerLookups(writer, set, term.lhs, fuel - 1);
            try producerLookups(writer, set, term.rhs, fuel - 1);
        },
        .unary => |term| try producerLookups(writer, set, term.operand, fuel - 1),
        .refinement_predicate => |term| {
            try producerLookups(writer, set, term.value, fuel - 1);
            for (term.args) |arg| try producerLookups(writer, set, arg, fuel - 1);
        },
        .bool_lit, .int_lit, .variable, .place_read, .invocation_caller => {},
        else => return error.UnsupportedLifecycleFormula,
    }
}
