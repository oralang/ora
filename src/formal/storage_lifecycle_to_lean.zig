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
    try writer.writeAll("import Ora.Storage.LifecycleAdapter\n");
    try writeProjectionNamed(writer, allocator, inputs.inventory, set, inputs.functions, "Ora.Generated.Obligations.Lifecycle");
    try writer.print("\nnamespace Ora.Generated.Obligations.Lifecycle\nopen Ora.Obligation Ora.Loop Ora.StorageLifecycle\ndef lifecycleInputsSha256 : String := \"{s}\"\n", .{std.fmt.bytesToHex(try inputs.sha256(allocator), .lower)});
    for (inputs.inventory.claims, 0..) |claim, ci| {
        const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
        try writer.print("namespace LC{d}\ndef word (bits : BitVec {d}) : Value := .integer {{shape := .unsigned .w{d}, bits}}\ndef claim : Claim := {{location := {{root := ", .{ ci, info.width, info.width });
        try string(writer, claim.root);
        try writer.print("}}, name := \"{s}\", bounds := [", .{@tagName(claim.bounds.?.kind)});
        if (claim.bounds.?.kind != .MaxValue) try value(writer, info, claim.bounds.?.minimum);
        if (claim.bounds.?.kind == .InRange) try writer.writeAll(", ");
        if (claim.bounds.?.kind != .MinValue) try value(writer, info, claim.bounds.?.maximum);
        try writer.writeAll("]}\ndef semantics := (adapterSemantics {} (directStorageContext emittedManifest Env.empty [])).withCallChecks\n");
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
            try writer.writeAll(
                \\def root : Root (CallFormula AdapterFormula FormulaRef AdapterChild) StorageLocationExpr FormulaRef AdapterChild := {semantics, events, entry := fun state =>
                \\  ∃ input, (row.callSignature.entry? input = some state)}
                \\def target : Prop := row.paths.length = 1 ∧
                \\  emittedCallManifest.timeline {} (row.paths[0]'(by decide)).timeline = .ok original ∧
                \\  successfulView original selected = some events ∧ root.Checked [claim] ∧
                \\  claim.holds (word 0) ∧ ∃ input state result,
                \\    row.callSignature.entry? input = some state ∧ Invariant [claim] state.loop.storage ∧
                \\    TimelineExec semantics strictEvents state result
                \\
            );
            try writer.print("end F{d}\n", .{row.id});
        }
        try writer.writeAll("def roots := [");
        for (set.function_summaries, 0..) |row, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("F{d}.root", .{row.id});
        }
        try writer.writeAll("]\ndef initial (store : StorageState) : Prop := store claim.location = word 0\ndef certificateTarget : Prop := Certificate [claim] initial (roots.map Root.step)\n");
        try writer.print("end LC{d}\n", .{ci});
    }
    try writer.writeAll("def emittedLifecycle : Prop := ");
    for (inputs.inventory.claims, 0..) |_, ci| {
        for (set.function_summaries) |row| try writer.print("LC{d}.F{d}.target ∧ ", .{ ci, row.id });
        try writer.print("LC{d}.certificateTarget ∧ ", .{ci});
    }
    try writer.writeAll("True\nend Ora.Generated.Obligations.Lifecycle\n");
}

fn writerRecipe(claim: data.Claim, row: obligation.FunctionSummaryRow, inputs: data.ProofInputs) !bool {
    if (row.paths.len != 1) return error.UnsupportedLifecycleRoot;
    var writes: usize = 0;
    for (row.paths[0].timeline, 0..) |event, index| {
        if (event.payload != .storage_write) continue;
        const write = event.payload.storage_write;
        if (!std.mem.eql(u8, write.location.root, claim.root)) continue;
        writes += 1;
        if (index != 1 or row.parameters.len != 1 or !obligation.typeRefEqual(row.parameters[0].ty, .{ .compiler_type_id = claim.type_id.? }))
            return error.UnsupportedLifecycleWriter;
        for (inputs.guards) |guard| {
            if (guard.function_id == row.id and guard.event_id == 0) break;
        } else return error.MissingLifecycleRetainedGuard;
    }
    if (writes > 1) return error.UnsupportedLifecycleWriter;
    return writes == 1;
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
    try writer.writeAll("\nnamespace Ora.Generated.Obligations.Lifecycle\nopen Ora.Obligation Ora.Loop Ora.StorageLifecycle\n");
    for (inputs.inventory.claims, 0..) |claim, ci| {
        const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
        try writer.print("namespace LC{d}\ntheorem zero_valid : claim.holds (word 0) := by\n  change ", .{ci});
        if (claim.bounds.?.kind == .InRange) try writer.writeAll("(0 : Nat) ≤ 0 ∧ ");
        if (claim.bounds.?.kind == .MinValue) try writer.writeAll("(0 : Nat) ≤ 0") else try writer.print("(0 : Nat) ≤ {d}", .{claim.bounds.?.maximum});
        try writer.writeAll("\n  decide\n");
        for (set.function_summaries) |row| {
            try writer.print("namespace F{d}\n", .{row.id});
            try writer.writeAll("attribute [local irreducible] emittedManifest\n");
            const guarded = try writerRecipe(claim, row, inputs);
            if (guarded) {
                const parameter = row.parameters[0];
                const identity = parameter.id orelse return error.UnsupportedLifecycleParameter;
                const checkpoint = row.paths[0].timeline[0].payload.safety_checkpoint.formulas[0].formula.term;
                const written = row.paths[0].timeline[1].payload.storage_write.value.formula.term;
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
                try writer.writeAll("theorem preserved : root.Checked [claim] := by\n  intro state ⟨input, entry⟩ valid\n  obtain ⟨bits, rfl⟩ := entry_shape input state entry\n");
                try writer.print("  apply guarded_write_prefixes semantics (.ordinary (.inl (.term {d}))) valid (value := word bits)\n    (after := {{state := {{entryState bits input.storage with loop := {{locals := [], storage := input.storage.set claim.location (word bits)}}}}, writeTrace := [claim.location]}})\n", .{checkpoint});
                try writer.writeAll("  ·\n");
                try producerFormulaBridge(writer, set, checkpoint, "    ");
                try writer.writeAll("  · ");
                try writer.print("change execStorageAction? semantics.storage (.write {{root := ", .{});
                try string(writer, claim.root);
                try writer.print(", projections := [], valueTy := .compilerTypeId {d}}} (.term {d})) _ = _\n", .{ claim.type_id.?, written });
                try producerWriteBridge(writer, set, written, "    ");
                try writer.writeAll("  · rfl\n  ·\n");
                try producerFrames(writer, row.paths[0].timeline[2..], "    ");
            } else {
                try writer.writeAll(
                    \\theorem preserved : root.Checked [claim] := by
                    \\  intro state _ valid
                    \\  apply framed_prefixes _ valid
                    \\  intro observed member
                    \\  have equal : observed = claim := by simpa using member
                    \\  subst observed
                    \\
                );
                try producerFrames(writer, row.paths[0].timeline, "  ");
            }
            try writer.writeAll("def witnessState : StorageActionState := {loop := {locals := [], storage := fun _ => word 0}, entryStorage := fun _ => word 0, reads := [");
            for (row.parameters, 0..) |parameter, index| {
                if (index != 0) try writer.writeAll(", ");
                const identity = parameter.id orelse return error.UnsupportedLifecycleParameter;
                try writer.print("{{id := {{file_id := {d}, pattern_id := {d}}}, value := ", .{ identity.file_id, identity.pattern_id });
                try parameterZero(writer, parameter.ty);
                try writer.writeByte('}');
            }
            try writer.writeAll("]}\ndef witnessInput : CallInput := {storage := fun _ => word 0, arguments := [");
            for (row.parameters, 0..) |parameter, index| {
                if (index != 0) try writer.writeAll(", ");
                try parameterZero(writer, parameter.ty);
            }
            try writer.writeAll("]}\ntheorem witnessed : ∃ result, TimelineExec semantics strictEvents witnessState result := by\n  refine ⟨_, ");
            for (row.paths[0].timeline) |event| {
                switch (event.payload) {
                    .safety_checkpoint => |checkpoint| {
                        try writer.writeAll(".cons (.assume (by\n");
                        try producerFormulaBridge(writer, set, checkpoint.formulas[0].formula.term, "    ");
                        try writer.writeAll("  ) (by try simp only [Ora.Integer.unsignedLe]\n        decide)) (");
                    },
                    .storage_write => |write| {
                        try writer.writeAll(".cons (.storage (by\n");
                        try producerWriteBridge(writer, set, write.value.formula.term, "    ");
                        try writer.writeAll("  )) (");
                    },
                    .storage_read => try writer.writeAll(".cons (.storage (by rfl)) ("),
                    else => return error.UnsupportedLifecycleEvent,
                }
            }
            try writer.writeAll(".nil _");
            for (row.paths[0].timeline) |_| try writer.writeByte(')');
            try writer.writeAll(
                \\⟩
                \\theorem proved : target := by
                \\  refine ⟨rfl, rfl, rfl, preserved, zero_valid, witnessInput, witnessState, ?_⟩
                \\  obtain ⟨result, trace⟩ := witnessed
                \\  refine ⟨result, rfl, ?_, trace⟩
                \\  intro observed member
                \\  have equal : observed = claim := by simpa using member
                \\  subst observed
                \\  exact zero_valid
                \\
            );
            try writer.print("end F{d}\n", .{row.id});
        }
        try writer.writeAll(
            \\theorem certificateProof : certificateTarget := by
            \\  apply prefix_certificate (claims := [claim]) (initial := initial) roots ⟨fun _ => word 0, rfl⟩
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
        for (set.function_summaries, 0..) |_, index| try writer.writeAll(if (index == 0) "rfl" else " | rfl");
        try writer.writeByte('\n');
        for (set.function_summaries) |row| try writer.print("    · exact F{d}.preserved\n", .{row.id});
        try writer.print("end LC{d}\n", .{ci});
    }
    try writer.writeAll("theorem lifecycleProof : emittedLifecycle := by\n  exact ⟨");
    for (inputs.inventory.claims, 0..) |_, ci| {
        for (set.function_summaries) |row| try writer.print("LC{d}.F{d}.proved, ", .{ ci, row.id });
        try writer.print("LC{d}.certificateProof, ", .{ci});
    }
    try writer.writeAll("True.intro⟩\nend Ora.Generated.Obligations.Lifecycle\n");
}

fn parameterZero(writer: anytype, ty: obligation.TypeRef) !void {
    if (ty != .compiler_type_id) return error.UnsupportedLifecycleParameter;
    const info = builtin.integerInfoByComptimeTypeId(ty.compiler_type_id) orelse return error.UnsupportedLifecycleParameter;
    try value(writer, info, 0);
}

fn producerFrames(writer: anytype, events: []const obligation.LoopTimelineEvent, indent: []const u8) !void {
    try writer.print("{s}intro event belongs\n", .{indent});
    if (events.len == 0) return writer.print("{s}simp only [root, events, List.not_mem_nil] at belongs\n", .{indent});
    try writer.print("{s}simp only [root, events, List.mem_cons, List.not_mem_nil, or_false] at belongs\n", .{indent});
    try writer.print("{s}rcases belongs with ", .{indent});
    for (events, 0..) |_, index| try writer.writeAll(if (index == 0) "rfl" else " | rfl");
    try writer.writeByte('\n');
    for (events) |event| {
        if (event.payload != .storage_write) {
            try writer.print("{s}· trivial\n", .{indent});
            continue;
        }
        const location = event.payload.storage_write.location;
        try writer.print("{s}· intro state location evaluates\n{s}  have same := direct_location (directStorageContext emittedManifest Env.empty []) ", .{ indent, indent });
        try string(writer, location.root);
        try writer.print(" (.compilerTypeId {d}) state location evaluates\n{s}  rw [same]\n{s}  decide\n", .{ location.value_ty.compiler_type_id, indent, indent });
    }
}

fn producerFormulaBridge(writer: anytype, set: obligation.ObligationSet, term: obligation.TermId, indent: []const u8) !void {
    try writer.print("{s}change (directStorageContext emittedManifest Env.empty []).evalFormula? (.term {d}) _ = _\n{s}unfold StorageFormulaContext.evalFormula?\n{s}rw [direct_empty_env _ _ (by rfl)]\n", .{ indent, term, indent, indent });
    try writer.print("{s}simp only [Option.bind_eq_bind, Option.some_bind, StorageFormulaContext.formulaStorageReady, formulaDenotes?, emitted_terms_length]\n{s}simp only [directStorageContext, emitted_terms_length, StorageFormulaContext.termStorageReady, ", .{ indent, indent });
    try producerLookups(writer, set, term, set.terms.len + 1);
    try writer.print("Bool.and_true, Bool.not_true, Bool.false_eq_true, if_false, Option.pure_def, Option.some_bind]\n{s}simp only [emitted_terms_length, denoteFormula?, denoteValue?, ", .{indent});
    try producerLookups(writer, set, term, set.terms.len + 1);
    try writer.print("Option.some_bind]\n{s}rfl\n", .{indent});
}

fn producerWriteBridge(writer: anytype, set: obligation.ObligationSet, term: obligation.TermId, indent: []const u8) !void {
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
        .bool_lit, .int_lit, .variable, .place_read => {},
        else => return error.UnsupportedLifecycleFormula,
    }
}
