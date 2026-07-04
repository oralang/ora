//! Canonical obligation manifest to Lean emitter.
//!
//! This emitter consumes the same `obligation.ObligationSet` used by the first
//! Z3 adapter and writes a Lean module containing manifest data, semantic
//! proposition names, and a kernel-checked structural well-formedness theorem.

const std = @import("std");
const obligation = @import("obligation.zig");
const type_builtin = @import("ora_types").builtin;

pub const Options = struct {
    namespace: []const u8 = "Ora.Generated.ObligationSmoke",
};

pub const SemanticSupport = union(enum) {
    supported,
    unsupported: SemanticUnsupportedReason,
};

pub const SemanticUnsupportedReason = union(enum) {
    empty_query,
    invalid_dependency,
    unsupported_obligation_kind,
    unsupported_effect_frame_relation: obligation.EffectFrameRelation,
    unsupported_origin_value,
    unsupported_term_kind,
    missing_type,
    unsupported_type: obligation.TypeRef,
    unsupported_comparison_width,
    unknown_signedness,
    mixed_signedness,
};

pub fn writeDataModule(writer: anytype, set: obligation.ObligationSet, options: Options) !void {
    try set.validateTermReferences();
    try set.validateIdReferences();

    try writeModulePreamble(writer, "Ora.Obligation.Manifest", options.namespace);
    try writeManifestRows(writer, set);
    try writeManifestDefinition(writer);
    try writeRowDefinitions(writer, set);
    try writeNamespaceEnd(writer, options.namespace);
}

pub fn writeModule(writer: anytype, set: obligation.ObligationSet, options: Options) !void {
    try set.validateTermReferences();
    try set.validateIdReferences();

    try writeModulePreamble(writer, "Ora.Obligation.Semantics", options.namespace);
    try writeManifestRows(writer, set);
    try writeManifestDefinition(writer);
    try writer.writeAll("theorem emitted_manifest_wf : emittedManifest.wf = true := by decide\n\n");
    try writeSemanticDefinitions(writer, set);
    try writeNamespaceEnd(writer, options.namespace);
}

fn writeModulePreamble(writer: anytype, import_name: []const u8, namespace: []const u8) !void {
    try writer.writeAll("import ");
    try writer.writeAll(import_name);
    try writer.writeAll("\n\nnamespace ");
    try writer.writeAll(namespace);
    try writer.writeAll("\n\nopen Ora.Obligation\n\n");
}

fn writeManifestRows(writer: anytype, set: obligation.ObligationSet) !void {
    try writeTerms(writer, set.terms);
    try writer.writeByte('\n');
    try writeAssumptions(writer, set.assumptions);
    try writer.writeByte('\n');
    try writeObligations(writer, set.obligations);
    try writer.writeByte('\n');
    try writeProofArtifacts(writer, set.proof_artifacts);
    try writer.writeByte('\n');
}

fn writeManifestDefinition(writer: anytype) !void {
    try writer.writeAll("def emittedManifest : Manifest := {\n");
    try writer.writeAll("  terms := emittedTerms,\n");
    try writer.writeAll("  assumptions := emittedAssumptions,\n");
    try writer.writeAll("  obligations := emittedObligations,\n");
    try writer.writeAll("  proofArtifacts := emittedProofArtifacts\n");
    try writer.writeAll("}\n\n");
}

fn writeNamespaceEnd(writer: anytype, namespace: []const u8) !void {
    try writer.writeAll("end ");
    try writer.writeAll(namespace);
    try writer.writeByte('\n');
}

fn writeTerms(writer: anytype, terms: []const obligation.Term) !void {
    try writer.writeAll("def emittedTerms : List Term := ");
    if (terms.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (terms, 0..) |term, index| {
        try writer.writeAll("  ");
        try writeTerm(writer, term);
        try writeListSeparator(writer, index, terms.len);
    }
    try writer.writeAll("]\n");
}

fn writeAssumptions(writer: anytype, rows: []const obligation.Assumption) !void {
    try writer.writeAll("def emittedAssumptions : List AssumptionRow := ");
    if (rows.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (rows, 0..) |row, index| {
        try writer.writeAll("  ");
        try writeAssumptionRow(writer, row);
        try writeListSeparator(writer, index, rows.len);
    }
    try writer.writeAll("]\n");
}

fn writeObligations(writer: anytype, rows: []const obligation.Obligation) !void {
    try writer.writeAll("def emittedObligations : List ObligationRow := ");
    if (rows.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (rows, 0..) |row, index| {
        try writer.writeAll("  ");
        try writeObligationRow(writer, row);
        try writeListSeparator(writer, index, rows.len);
    }
    try writer.writeAll("]\n");
}

fn writeAssumptionRow(writer: anytype, row: obligation.Assumption) !void {
    try writer.writeAll("{ id := ");
    try writer.print("{d}", .{row.id});
    try writer.writeAll(", owner := ");
    try writeLeanString(writer, ownerName(row.owner));
    try writer.writeAll(", kind := .");
    try writer.writeAll(assumptionKindName(row.kind));
    try writer.writeAll(", formula := ");
    if (row.formula) |formula| {
        try writer.writeAll("some (");
        try writeFormula(writer, formula);
        try writer.writeByte(')');
    } else {
        try writer.writeAll("none");
    }
    try writer.writeAll(" }");
}

fn writeObligationRow(writer: anytype, row: obligation.Obligation) !void {
    try writer.writeAll("{ id := ");
    try writer.print("{d}", .{row.id});
    try writer.writeAll(", owner := ");
    try writeLeanString(writer, ownerName(row.owner));
    try writer.writeAll(", kind := ");
    try writeObligationKind(writer, row.kind);
    try writer.writeAll(" }");
}

fn writeSemanticDefinitions(writer: anytype, set: obligation.ObligationSet) !void {
    var emitted_any = false;

    try writeRowDefinitions(writer, set);

    for (set.obligations) |row| {
        switch (kindSemanticSupport(set, row.kind)) {
            .supported => {},
            .unsupported => continue,
        }
        if (!emitted_any) {
            try writer.writeAll("-- Semantic proposition names for proof attachment.\n");
            emitted_any = true;
        }
        try writer.writeAll("def emittedObligation_");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(" : Prop :=\n  obligationDenotes emittedManifest ");
        try writeObligationRow(writer, row);
        try writer.writeAll("\n\n");
    }

    for (set.queries) |query| {
        if (!try querySupportsSemantics(set, query)) continue;
        if (!emitted_any) {
            try writer.writeAll("-- Semantic proposition names for proof attachment.\n");
            emitted_any = true;
        }
        try writer.writeAll("def emittedQuery_");
        try writer.print("{d}", .{query.id});
        try writer.writeAll(" : Prop :=\n");
        for (query.obligation_ids, 0..) |obligation_id, index| {
            const row = findObligation(set, obligation_id) orelse return error.InvalidDependency;
            if (index != 0) try writer.writeAll(" /\\\n");
            try writer.writeAll("  obligationFollowsFromAssumptions emittedManifest\n    ");
            try writeAssumptionRowsByIds(writer, set, query.assumption_ids);
            try writer.writeAll("\n    ");
            try writeObligationRow(writer, row);
        }
        try writer.writeAll("\n\n");
    }
}

fn writeRowDefinitions(writer: anytype, set: obligation.ObligationSet) !void {
    for (set.assumptions) |row| {
        try writer.writeAll("def emittedAssumptionRow_");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(" : AssumptionRow :=\n  ");
        try writeAssumptionRow(writer, row);
        try writer.writeAll("\n\n");
    }

    for (set.obligations) |row| {
        try writer.writeAll("def emittedObligationRow_");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(" : ObligationRow :=\n  ");
        try writeObligationRow(writer, row);
        try writer.writeAll("\n\n");
    }
}

fn writeAssumptionRowsByIds(writer: anytype, set: obligation.ObligationSet, ids: []const obligation.Id) !void {
    if (ids.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        const row = findAssumption(set, id) orelse return error.InvalidDependency;
        try writeAssumptionRow(writer, row);
    }
    try writer.writeByte(']');
}

fn findObligation(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Obligation {
    for (set.obligations) |row| {
        if (row.id == id) return row;
    }
    return null;
}

fn findAssumption(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Assumption {
    for (set.assumptions) |row| {
        if (row.id == id) return row;
    }
    return null;
}

pub fn querySemanticSupport(set: obligation.ObligationSet, query: obligation.VerificationQuery) SemanticSupport {
    if (query.obligation_ids.len == 0) return .{ .unsupported = .empty_query };

    for (query.assumption_ids) |assumption_id| {
        const row = findAssumption(set, assumption_id) orelse return .{ .unsupported = .invalid_dependency };
        if (row.formula) |formula| {
            switch (formulaSemanticSupport(set, formula)) {
                .supported => {},
                .unsupported => |reason| return .{ .unsupported = reason },
            }
        } else {
            return .{ .unsupported = .unsupported_origin_value };
        }
    }

    for (query.obligation_ids) |obligation_id| {
        const row = findObligation(set, obligation_id) orelse return .{ .unsupported = .invalid_dependency };
        switch (kindSemanticSupport(set, row.kind)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }

    return .supported;
}

fn querySupportsSemantics(set: obligation.ObligationSet, query: obligation.VerificationQuery) !bool {
    return switch (querySemanticSupport(set, query)) {
        .supported => true,
        .unsupported => |reason| switch (reason) {
            .invalid_dependency => error.InvalidDependency,
            else => false,
        },
    };
}

fn kindSemanticSupport(set: obligation.ObligationSet, kind: obligation.Kind) SemanticSupport {
    return switch (kind) {
        .logical => |logical| formulaSemanticSupport(set, logical.formula),
        .runtime_guard => |guard| formulaSemanticSupport(set, guard.formula),
        .effect_frame => |effect| effectFrameSemanticSupport(effect),
        .type_wf,
        .type_relation,
        .region_relation,
        .resource,
        .quantifier,
        .filtered_input,
        .backend_fact,
        => .{ .unsupported = .unsupported_obligation_kind },
    };
}

fn effectFrameSemanticSupport(effect: obligation.EffectFrameGoal) SemanticSupport {
    return switch (effect.relation) {
        .write_covered_by_modifies,
        .read_preserved_by_frame,
        => .supported,
        .lock_covers_write,
        .external_call_frame,
        => .{ .unsupported = .{ .unsupported_effect_frame_relation = effect.relation } },
    };
}

fn formulaSemanticSupport(set: obligation.ObligationSet, formula: obligation.FormulaRef) SemanticSupport {
    return switch (formula) {
        .term => |id| formulaTermSemanticSupport(set, id, set.terms.len + 1),
        .origin_value => .{ .unsupported = .unsupported_origin_value },
    };
}

fn termById(set: obligation.ObligationSet, id: obligation.TermId) ?obligation.Term {
    if (id >= set.terms.len) return null;
    return set.terms[id];
}

fn formulaTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .bool_lit => .supported,
        .variable => |variable| varRefSupportsBoolFormula(variable),
        .refinement_predicate => |predicate| refinementPredicateSemanticSupport(set, predicate, fuel - 1),
        .unary => |unary| switch (unary.op) {
            .not => formulaTermSemanticSupport(set, unary.operand, fuel - 1),
            .neg => .{ .unsupported = .unsupported_term_kind },
        },
        .binary => |binary| binaryFormulaSemanticSupport(set, binary, fuel - 1),
        .quantified => |quantified| quantifiedSemanticSupport(set, quantified, fuel - 1),
        .int_lit,
        .old,
        .result,
        .place_read,
        => .{ .unsupported = .unsupported_term_kind },
    };
}

fn valueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .bool_lit => .supported,
        .int_lit => |literal| optionalTypeSupportsU256Carrier(literal.ty),
        .variable => |variable| varRefSupportsBoolOrU256CarrierValue(variable),
        .result => .supported,
        .place_read => .supported,
        .binary => |binary| switch (binary.op) {
            .add, .sub => binaryU256ValueSemanticSupport(set, binary, fuel - 1),
            else => .{ .unsupported = .unsupported_term_kind },
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        .unary,
        .refinement_predicate,
        .quantified,
        => .{ .unsupported = .unsupported_term_kind },
    };
}

fn u256ValueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .int_lit => |literal| optionalTypeSupportsU256(literal.ty),
        .variable => |variable| varRefSupportsU256Value(variable),
        .result => .supported,
        .place_read => .supported,
        .binary => |binary| switch (binary.op) {
            .add, .sub => binaryU256ValueSemanticSupport(set, binary, fuel - 1),
            else => .{ .unsupported = .unsupported_term_kind },
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn oldPlaceReadSemanticSupport(set: obligation.ObligationSet, operand: obligation.TermId) SemanticSupport {
    const term = termById(set, operand) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .place_read => .supported,
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn binaryFormulaSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    return switch (binary.op) {
        .eq, .ne => firstUnsupported(.{
            valueTermSemanticSupport(set, binary.lhs, fuel),
            valueTermSemanticSupport(set, binary.rhs, fuel),
        }),
        .lt, .le, .gt, .ge => firstUnsupported(.{
            u256ValueTermSemanticSupport(set, binary.lhs, fuel),
            u256ValueTermSemanticSupport(set, binary.rhs, fuel),
        }),
        .slt, .sle, .sgt, .sge => signedComparisonSemanticSupport(set, binary, fuel),
        .and_, .or_, .implies => firstUnsupported(.{
            formulaTermSemanticSupport(set, binary.lhs, fuel),
            formulaTermSemanticSupport(set, binary.rhs, fuel),
        }),
        .add,
        .sub,
        .mul,
        .div,
        .mod,
        => .{ .unsupported = .unsupported_term_kind },
    };
}

fn i256ValueTermSemanticSupport(
    set: obligation.ObligationSet,
    id: obligation.TermId,
    fuel: usize,
) SemanticSupport {
    if (fuel == 0) return .{ .unsupported = .unsupported_term_kind };
    const term = termById(set, id) orelse return .{ .unsupported = .invalid_dependency };
    return switch (term) {
        .int_lit => |literal| optionalTypeSupportsI256(literal.ty),
        .variable => |variable| varRefSupportsI256Value(variable),
        .result => .supported,
        .place_read => .supported,
        .binary => |binary| switch (binary.op) {
            .add, .sub => binaryI256ValueSemanticSupport(set, binary, fuel - 1),
            else => .{ .unsupported = .unsupported_term_kind },
        },
        .old => |operand| oldPlaceReadSemanticSupport(set, operand),
        else => .{ .unsupported = .unsupported_term_kind },
    };
}

fn signedComparisonSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    switch (optionalTypeSupportsI256(binary.ty)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }
    return firstUnsupported(.{
        i256ValueTermSemanticSupport(set, binary.lhs, fuel),
        i256ValueTermSemanticSupport(set, binary.rhs, fuel),
    });
}

fn binaryI256ValueSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    switch (optionalTypeSupportsI256(binary.ty)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }
    return firstUnsupported(.{
        i256ValueTermSemanticSupport(set, binary.lhs, fuel),
        i256ValueTermSemanticSupport(set, binary.rhs, fuel),
    });
}

fn binaryU256ValueSemanticSupport(
    set: obligation.ObligationSet,
    binary: obligation.BinaryTerm,
    fuel: usize,
) SemanticSupport {
    return firstUnsupported(.{
        u256ValueTermSemanticSupport(set, binary.lhs, fuel),
        u256ValueTermSemanticSupport(set, binary.rhs, fuel),
    });
}

fn refinementPredicateSemanticSupport(
    set: obligation.ObligationSet,
    predicate: obligation.RefinementPredicateTerm,
    fuel: usize,
) SemanticSupport {
    switch (u256ValueTermSemanticSupport(set, predicate.value, fuel)) {
        .supported => {},
        .unsupported => |reason| return .{ .unsupported = reason },
    }
    for (predicate.args) |arg| {
        switch (u256ValueTermSemanticSupport(set, arg, fuel)) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return .supported;
}

fn quantifiedSemanticSupport(
    set: obligation.ObligationSet,
    quantified: obligation.QuantifiedTerm,
    fuel: usize,
) SemanticSupport {
    var binder_reason: ?SemanticUnsupportedReason = null;
    switch (optionalTypeSupportsU256Carrier(quantified.binder.ty)) {
        .supported => {},
        .unsupported => |reason| binder_reason = reason,
    }
    if (quantified.condition) |condition| {
        switch (formulaTermSemanticSupport(set, condition, fuel)) {
            .supported => {},
            .unsupported => |reason| {
                if (reasonIsSignedComparisonTypeGate(reason)) return .{ .unsupported = reason };
                return .{ .unsupported = binder_reason orelse reason };
            },
        }
    }
    switch (formulaTermSemanticSupport(set, quantified.body, fuel)) {
        .supported => {},
        .unsupported => |reason| {
            if (reasonIsSignedComparisonTypeGate(reason)) return .{ .unsupported = reason };
            return .{ .unsupported = binder_reason orelse reason };
        },
    }
    if (binder_reason) |reason| return .{ .unsupported = reason };
    return .supported;
}

fn firstUnsupported(results: anytype) SemanticSupport {
    inline for (results) |result| {
        switch (result) {
            .supported => {},
            .unsupported => |reason| return .{ .unsupported = reason },
        }
    }
    return .supported;
}

fn reasonIsSignedComparisonTypeGate(reason: SemanticUnsupportedReason) bool {
    return switch (reason) {
        .unsupported_comparison_width,
        .unknown_signedness,
        .mixed_signedness,
        => true,
        else => false,
    };
}

fn varRefSupportsBoolFormula(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsBool(free.ty),
        .bound => |bound| optionalTypeSupportsBool(bound.ty),
    };
}

fn varRefSupportsBoolOrU256CarrierValue(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsBoolOrU256Carrier(free.ty),
        .bound => |bound| optionalTypeSupportsBoolOrU256Carrier(bound.ty),
    };
}

fn varRefSupportsU256Value(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsU256(free.ty),
        .bound => |bound| optionalTypeSupportsU256(bound.ty),
    };
}

fn varRefSupportsI256Value(variable: obligation.VarRef) SemanticSupport {
    return switch (variable) {
        .free => |free| optionalTypeSupportsI256(free.ty),
        .bound => |bound| optionalTypeSupportsI256(bound.ty),
    };
}

fn optionalTypeSupportsBool(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsBool(value)) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn optionalTypeSupportsU256(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsU256(value)) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn optionalTypeSupportsI256(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .unknown_signedness };
    const info = compilerIntegerInfo(value) orelse return .{ .unsupported = .unknown_signedness };
    if (info.width != 256) return .{ .unsupported = .unsupported_comparison_width };
    if (!info.signed) return .{ .unsupported = .mixed_signedness };
    return .supported;
}

fn optionalTypeSupportsU256Carrier(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsU256(value) or typeRefIsI256(value)) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn optionalTypeSupportsBoolOrU256Carrier(ty: ?obligation.TypeRef) SemanticSupport {
    const value = ty orelse return .{ .unsupported = .missing_type };
    if (typeRefIsBool(value) or typeRefIsU256(value) or typeRefIsI256(value)) return .supported;
    return .{ .unsupported = .{ .unsupported_type = value } };
}

fn typeRefIsBool(ty: obligation.TypeRef) bool {
    return switch (ty) {
        .spelling => |name| std.mem.eql(u8, name, "bool") or std.mem.eql(u8, name, "i1"),
        .compiler_type_id => |id| compilerTypeIdIsBuiltin(id, .bool),
    };
}

fn typeRefIsU256(ty: obligation.TypeRef) bool {
    return switch (ty) {
        .spelling => |name| std.mem.eql(u8, name, "u256") or std.mem.eql(u8, name, "uint256"),
        .compiler_type_id => |id| compilerTypeIdIsBuiltin(id, .u256),
    };
}

fn typeRefIsI256(ty: obligation.TypeRef) bool {
    return switch (ty) {
        .spelling => |name| std.mem.eql(u8, name, "i256") or std.mem.eql(u8, name, "int256"),
        .compiler_type_id => |id| compilerTypeIdIsBuiltin(id, .i256),
    };
}

const CompilerIntegerInfo = struct {
    width: u32,
    signed: bool,
};

fn compilerIntegerInfo(ty: obligation.TypeRef) ?CompilerIntegerInfo {
    return switch (ty) {
        .compiler_type_id => |id| blk: {
            const info = type_builtin.integerInfoByComptimeTypeId(id) orelse break :blk null;
            break :blk .{
                .width = info.width,
                .signed = info.signed,
            };
        },
        .spelling => null,
    };
}

fn compilerTypeIdIsBuiltin(id: u32, builtin: type_builtin.BuiltinTypeId) bool {
    return id == type_builtin.lookupBuiltinById(builtin).comptime_type_id;
}

fn writeProofArtifacts(writer: anytype, rows: []const obligation.ProofArtifact) !void {
    try writer.writeAll("def emittedProofArtifacts : List ProofArtifactRow := ");
    if (rows.len == 0) return writer.writeAll("[]\n");

    try writer.writeAll("[\n");
    for (rows, 0..) |row, index| {
        try writer.writeAll("  { id := ");
        try writer.print("{d}", .{row.id});
        try writer.writeAll(", owner := ");
        try writeLeanString(writer, ownerName(row.owner));
        try writer.writeAll(", kind := .");
        try writer.writeAll(proofArtifactKindName(row.kind));
        try writer.writeAll(", moduleName := ");
        try writeLeanString(writer, row.module_name);
        try writer.writeAll(", theoremName := ");
        try writeLeanString(writer, row.theorem_name);
        try writer.writeAll(", path := ");
        try writeOptionalLeanString(writer, row.path);
        try writer.writeAll(", contentHash := ");
        if (row.content_hash) |hash| {
            try writer.writeAll("some ");
            try writer.print("{d}", .{hash});
        } else {
            try writer.writeAll("none");
        }
        try writer.writeAll(", obligationIds := ");
        try writeIdList(writer, row.obligation_ids);
        try writer.writeAll(" }");
        try writeListSeparator(writer, index, rows.len);
    }
    try writer.writeAll("]\n");
}

fn writeTerm(writer: anytype, term: obligation.Term) !void {
    switch (term) {
        .bool_lit => |value| {
            try writer.writeAll(".boolLit ");
            try writer.writeAll(if (value) "true" else "false");
        },
        .int_lit => |literal| {
            try writer.writeAll(".intLit { value := ");
            try writeLeanString(writer, literal.value);
            try writer.writeAll(", ty := ");
            try writeOptionalTypeRef(writer, literal.ty);
            try writer.writeAll(" }");
        },
        .variable => |variable| {
            try writer.writeAll(".variable (");
            try writeVarRef(writer, variable);
            try writer.writeByte(')');
        },
        .old => |id| {
            try writer.writeAll(".old ");
            try writer.print("{d}", .{id});
        },
        .result => try writer.writeAll(".result"),
        .place_read => |place| {
            try writer.writeAll(".placeRead ");
            try writePlaceRef(writer, place);
        },
        .unary => |unary| {
            try writer.writeAll(".unary { op := .");
            try writer.writeAll(unaryOpName(unary.op));
            try writer.writeAll(", operand := ");
            try writer.print("{d}", .{unary.operand});
            try writer.writeAll(" }");
        },
        .binary => |binary| {
            try writer.writeAll(".binary { op := .");
            try writer.writeAll(binaryOpName(binary.op));
            try writer.writeAll(", lhs := ");
            try writer.print("{d}", .{binary.lhs});
            try writer.writeAll(", rhs := ");
            try writer.print("{d}", .{binary.rhs});
            try writer.writeAll(", ty := ");
            try writeOptionalTypeRef(writer, binary.ty);
            try writer.writeAll(" }");
        },
        .refinement_predicate => |predicate| {
            try writer.writeAll(".refinementPredicate { name := ");
            try writeLeanString(writer, predicate.name);
            try writer.writeAll(", value := ");
            try writer.print("{d}", .{predicate.value});
            try writer.writeAll(", args := ");
            try writeTermIdList(writer, predicate.args);
            try writer.writeAll(" }");
        },
        .quantified => |quantified| {
            try writer.writeAll(".quantified { quantifier := .");
            try writer.writeAll(quantifierName(quantified.quantifier));
            try writer.writeAll(", binder := ");
            try writeBinderRef(writer, quantified.binder);
            try writer.writeAll(", condition := ");
            if (quantified.condition) |condition| {
                try writer.writeAll("some ");
                try writer.print("{d}", .{condition});
            } else {
                try writer.writeAll("none");
            }
            try writer.writeAll(", body := ");
            try writer.print("{d}", .{quantified.body});
            try writer.writeAll(" }");
        },
    }
}

fn writeFormula(writer: anytype, formula: obligation.FormulaRef) !void {
    switch (formula) {
        .term => |id| {
            try writer.writeAll(".term ");
            try writer.print("{d}", .{id});
        },
        .origin_value => return error.UnsupportedOriginValue,
    }
}

fn writeObligationKind(writer: anytype, kind: obligation.Kind) !void {
    switch (kind) {
        .logical => |logical| {
            try writer.writeAll(".logical .");
            try writer.writeAll(logicalRoleName(logical.role));
            try writer.writeAll(" (");
            try writeFormula(writer, logical.formula);
            try writer.writeByte(')');
        },
        .runtime_guard => |guard| {
            try writer.writeAll(".runtimeGuard ");
            try writeLeanString(writer, guard.guard_id);
            try writer.writeAll(" (");
            try writeFormula(writer, guard.formula);
            try writer.writeByte(')');
        },
        .effect_frame => |effect| {
            try writer.writeAll(".effectFrame ");
            try writeEffectFrameGoal(writer, effect);
        },
        .resource => |resource| {
            try writer.writeAll(".resource ");
            try writeResourceGoal(writer, resource);
        },
        .quantifier => |quantifier| {
            try writer.writeAll(".quantifier ");
            try writeQuantifierGoal(writer, quantifier);
        },
        .backend_fact => |fact| {
            try writer.writeAll(".backendFact ");
            try writeBackendFactGoal(writer, fact);
        },
        .type_wf,
        .type_relation,
        .region_relation,
        .filtered_input,
        => return error.UnsupportedObligationKind,
    }
}

fn writeBackendFactGoal(writer: anytype, fact: obligation.BackendFactGoal) !void {
    try writer.writeAll("{ component := .");
    try writer.writeAll(backendComponentName(fact.component));
    try writer.writeAll(", property := .");
    try writer.writeAll(backendPropertyName(fact.property));
    try writer.writeAll(" }");
}

fn writeQuantifierGoal(writer: anytype, quantifier: obligation.QuantifierGoal) !void {
    try writer.writeAll("{ quantifier := .");
    try writer.writeAll(quantifierName(quantifier.quantifier));
    try writer.writeAll(", binderName := ");
    try writeLeanString(writer, quantifier.variable);
    try writer.writeAll(", binderType := ");
    try writeTypeRef(writer, quantifier.binder_type);
    try writer.writeAll(", binderSort := .");
    try writer.writeAll(quantifierBinderSortName(quantifier.binder_sort));
    try writer.writeAll(", fragment := .");
    try writer.writeAll(queryFragmentName(quantifier.fragment));
    try writer.writeAll(", patternStatus := .");
    try writer.writeAll(quantifierPatternStatusName(quantifier.pattern_status));
    try writer.writeAll(", degradation := ");
    if (quantifier.degradation) |degradation| {
        try writer.writeAll("some .");
        try writer.writeAll(quantifierDegradationName(degradation));
    } else {
        try writer.writeAll("none");
    }
    try writer.writeAll(" }");
}

fn writeEffectFrameGoal(writer: anytype, effect: obligation.EffectFrameGoal) !void {
    try writer.writeAll("{ relation := .");
    try writer.writeAll(effectFrameRelationName(effect.relation));
    try writer.writeAll(", declared := ");
    try writePlaceRefList(writer, effect.declared);
    try writer.writeAll(", actual := ");
    try writePlaceRefList(writer, effect.actual);
    try writer.writeAll(" }");
}

fn writeResourceGoal(writer: anytype, resource: obligation.ResourceGoal) !void {
    try writer.writeAll("{ op := .");
    try writer.writeAll(resourceOperationName(resource.op));
    try writer.writeAll(", domain := ");
    try writeLeanString(writer, resource.domain);
    try writer.writeAll(", source := ");
    try writeOptionalPlaceRef(writer, resource.source);
    try writer.writeAll(", destination := ");
    try writeOptionalPlaceRef(writer, resource.destination);
    try writer.writeAll(", amount := ");
    if (resource.amount) |amount| {
        try writer.writeAll("some (");
        try writeFormula(writer, amount);
        try writer.writeByte(')');
    } else {
        try writer.writeAll("none");
    }
    try writer.writeAll(", property := .");
    try writer.writeAll(resourcePropertyName(resource.property));
    try writer.writeAll(" }");
}

fn writeOptionalPlaceRef(writer: anytype, place: ?obligation.PlaceRef) !void {
    if (place) |value| {
        try writer.writeAll("some ");
        try writePlaceRef(writer, value);
    } else {
        try writer.writeAll("none");
    }
}

fn writePlaceRefList(writer: anytype, places: []const obligation.PlaceRef) !void {
    if (places.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (places, 0..) |place, index| {
        if (index != 0) try writer.writeAll(", ");
        try writePlaceRef(writer, place);
    }
    try writer.writeByte(']');
}

fn writePlaceRef(writer: anytype, place: obligation.PlaceRef) !void {
    try writer.writeAll("{ root := ");
    try writeLeanString(writer, place.root);
    try writer.writeAll(", region := .");
    try writer.writeAll(regionName(place.region));
    try writer.writeAll(", fields := ");
    try writeStringList(writer, place.fields);
    try writer.writeAll(", keys := ");
    try writePlaceKeyList(writer, place.keys);
    try writer.writeAll(" }");
}

fn writeStringList(writer: anytype, values: []const []const u8) !void {
    if (values.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLeanString(writer, value);
    }
    try writer.writeByte(']');
}

fn writePlaceKeyList(writer: anytype, keys: []const obligation.PlaceKey) !void {
    if (keys.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (keys, 0..) |key, index| {
        if (index != 0) try writer.writeAll(", ");
        try writePlaceKey(writer, key);
    }
    try writer.writeByte(']');
}

fn writePlaceKey(writer: anytype, key: obligation.PlaceKey) !void {
    switch (key) {
        .parameter => |index| try writer.print(".parameter {d}", .{index}),
        .comptime_parameter => |index| try writer.print(".comptimeParameter {d}", .{index}),
        .comptime_range_parameter => |index| try writer.print(".comptimeRangeParameter {d}", .{index}),
        .constant => |value| {
            try writer.writeAll(".constant ");
            try writeLeanString(writer, value);
        },
        .msg_sender => try writer.writeAll(".msgSender"),
        .tx_origin => try writer.writeAll(".txOrigin"),
        .unknown => try writer.writeAll(".unknown"),
    }
}

fn writeVarRef(writer: anytype, variable: obligation.VarRef) !void {
    switch (variable) {
        .free => |free| {
            try writer.writeAll(".free ");
            try writeFreeVarRef(writer, free);
        },
        .bound => |bound| {
            try writer.writeAll(".bound ");
            try writeBoundVarRef(writer, bound);
        },
    }
}

fn writeFreeVarRef(writer: anytype, variable: obligation.FreeVarRef) !void {
    try writer.writeAll("{ id := { file_id := ");
    try writer.print("{d}", .{variable.id.file_id});
    try writer.writeAll(", pattern_id := ");
    try writer.print("{d}", .{variable.id.pattern_id});
    try writer.writeAll(" }");
    try writer.writeAll(", name := ");
    try writeLeanString(writer, variable.name);
    try writer.writeAll(", ty := ");
    try writeOptionalTypeRef(writer, variable.ty);
    try writer.writeAll(", region := ");
    try writeOptionalRegionRef(writer, variable.region);
    try writer.writeAll(" }");
}

fn writeBoundVarRef(writer: anytype, variable: obligation.BoundVarRef) !void {
    try writer.writeAll("{ index := ");
    try writer.print("{d}", .{variable.index});
    try writer.writeAll(", name := ");
    try writeLeanString(writer, variable.name);
    try writer.writeAll(", ty := ");
    try writeOptionalTypeRef(writer, variable.ty);
    try writer.writeAll(", region := ");
    try writeOptionalRegionRef(writer, variable.region);
    try writer.writeAll(" }");
}

fn writeBinderRef(writer: anytype, variable: obligation.BinderRef) !void {
    try writer.writeAll("{ name := ");
    try writeLeanString(writer, variable.name);
    try writer.writeAll(", ty := ");
    try writeOptionalTypeRef(writer, variable.ty);
    try writer.writeAll(", region := ");
    try writeOptionalRegionRef(writer, variable.region);
    try writer.writeAll(" }");
}

fn writeOptionalRegionRef(writer: anytype, region: ?obligation.RegionRef) !void {
    if (region) |value| {
        try writer.writeAll("some .");
        try writer.writeAll(regionName(value));
    } else {
        try writer.writeAll("none");
    }
}

fn writeOptionalTypeRef(writer: anytype, ty: ?obligation.TypeRef) !void {
    if (ty) |value| {
        try writer.writeAll("some (");
        try writeTypeRef(writer, value);
        try writer.writeByte(')');
    } else {
        try writer.writeAll("none");
    }
}

fn writeTypeRef(writer: anytype, ty: obligation.TypeRef) !void {
    switch (ty) {
        .spelling => |text| {
            try writer.writeAll(".spelling ");
            try writeLeanString(writer, text);
        },
        .compiler_type_id => |id| {
            try writer.writeAll(".compilerTypeId ");
            try writer.print("{d}", .{id});
        },
    }
}

fn writeTermIdList(writer: anytype, ids: []const obligation.TermId) !void {
    try writeIntegerList(writer, ids);
}

fn writeIdList(writer: anytype, ids: []const obligation.Id) !void {
    try writeIntegerList(writer, ids);
}

fn writeIntegerList(writer: anytype, ids: anytype) !void {
    if (ids.len == 0) return writer.writeAll("[]");
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeOptionalLeanString(writer: anytype, value: ?[]const u8) !void {
    if (value) |text| {
        try writer.writeAll("some ");
        try writeLeanString(writer, text);
    } else {
        try writer.writeAll("none");
    }
}

fn writeListSeparator(writer: anytype, index: usize, len: usize) !void {
    if (index + 1 != len) try writer.writeByte(',');
    try writer.writeByte('\n');
}

fn writeLeanString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    var start: usize = 0;
    for (value, 0..) |byte, index| {
        switch (byte) {
            '"', '\\', '\n', '\r', '\t', 0x00...0x08, 0x0b...0x0c, 0x0e...0x1f => {
                if (start < index) try writer.writeAll(value[start..index]);
                switch (byte) {
                    '"' => try writer.writeAll("\\\""),
                    '\\' => try writer.writeAll("\\\\"),
                    '\n' => try writer.writeAll("\\n"),
                    '\r' => try writer.writeAll("\\r"),
                    '\t' => try writer.writeAll("\\t"),
                    else => try writer.print("\\u{{{x}}}", .{byte}),
                }
                start = index + 1;
            },
            else => {},
        }
    }
    if (start < value.len) try writer.writeAll(value[start..]);
    try writer.writeByte('"');
}

fn ownerName(owner: obligation.Owner) []const u8 {
    return switch (owner) {
        .module => |name| name,
        .contract => |name| name,
        .function => |function| function.name,
        .trait_method => |method| method.method_name,
        .statement => |statement| statement.function_name,
        .backend => |backend| backend.name,
    };
}

fn regionName(region: obligation.RegionRef) []const u8 {
    return switch (region) {
        .none => "none",
        .storage => "storage",
        .memory => "memory",
        .transient => "transient",
        .calldata => "calldata",
    };
}

fn unaryOpName(op: obligation.UnaryOp) []const u8 {
    return switch (op) {
        .not => "not_",
        .neg => "neg",
    };
}

fn binaryOpName(op: obligation.BinaryOp) []const u8 {
    return switch (op) {
        .eq => "eq",
        .ne => "ne",
        .lt => "lt",
        .le => "le",
        .gt => "gt",
        .ge => "ge",
        .slt => "slt",
        .sle => "sle",
        .sgt => "sgt",
        .sge => "sge",
        .add => "add",
        .sub => "sub",
        .mul => "mul",
        .div => "div",
        .mod => "mod_",
        .and_ => "and_",
        .or_ => "or_",
        .implies => "implies",
    };
}

fn quantifierName(quantifier: obligation.Quantifier) []const u8 {
    return switch (quantifier) {
        .forall => "forall_",
        .exists => "exists_",
    };
}

fn queryFragmentName(fragment: obligation.VerificationQueryFragment) []const u8 {
    return switch (fragment) {
        .unknown => "unknown",
        .qf_bv => "qfBv",
        .qf_bv_array => "qfBvArray",
        .aufbv => "aufbv",
        .aufbv_quantifiers => "aufbvQuantifiers",
        .other => "other",
    };
}

fn quantifierBinderSortName(sort: obligation.QuantifierBinderSort) []const u8 {
    return switch (sort) {
        .bool => "bool",
        .bit_vector => "bitVector",
        .byte_sequence => "byteSequence",
        .opaque_unknown => "opaqueUnknown",
    };
}

fn quantifierPatternStatusName(status: obligation.QuantifierPatternStatus) []const u8 {
    return switch (status) {
        .explicit => "explicit",
        .synthesized => "synthesized",
        .absent => "absent",
    };
}

fn quantifierDegradationName(degradation: obligation.QuantifierDegradation) []const u8 {
    return switch (degradation) {
        .unsupported_binder_type => "unsupportedBinderType",
        .malformed_binder_width => "malformedBinderWidth",
    };
}

fn proofArtifactKindName(kind: obligation.ProofArtifactKind) []const u8 {
    return switch (kind) {
        .userland_lean => "userlandLean",
    };
}

fn backendComponentName(component: obligation.BackendComponent) []const u8 {
    return switch (component) {
        .dispatcher => "dispatcher",
        .oratosir => "oratosir",
        .sinora => "sinora",
        .artifact_policy => "artifactPolicy",
    };
}

fn backendPropertyName(property: obligation.BackendProperty) []const u8 {
    return switch (property) {
        .complete => "complete",
        .disjoint => "disjoint",
        .ordered => "ordered",
        .preserves_selector_behavior => "preservesSelectorBehavior",
        .no_unknown_strategy => "noUnknownStrategy",
        .dependency_valid => "dependencyValid",
    };
}

fn logicalRoleName(role: obligation.LogicalRole) []const u8 {
    return switch (role) {
        .invariant => "invariant",
        .requires => "requires",
        .callee_precondition => "calleePrecondition",
        .ensures => "ensures",
        .ensures_ok => "ensuresOk",
        .ensures_err => "ensuresErr",
        .assert => "assert_",
        .guard => "guard",
        .loop_invariant => "loopInvariant",
        .contract_invariant => "contractInvariant",
        .arithmetic_safety => "arithmeticSafety",
        .refinement => "refinement",
        .imported_callee_obligation => "importedCalleeObligation",
        .imported_callee_ensures => "importedCalleeEnsures",
    };
}

fn resourceOperationName(op: obligation.ResourceOperation) []const u8 {
    return switch (op) {
        .move => "move",
        .create => "create",
        .destroy => "destroy",
    };
}

fn resourcePropertyName(property: obligation.ResourceProperty) []const u8 {
    return switch (property) {
        .amount_non_negative => "amountNonNegative",
        .source_sufficient => "sourceSufficient",
        .destination_no_overflow => "destinationNoOverflow",
        .same_place_net_zero => "samePlaceNetZero",
        .conservation => "conservation",
        .modifies_covered => "modifiesCovered",
    };
}

fn effectFrameRelationName(relation: obligation.EffectFrameRelation) []const u8 {
    return switch (relation) {
        .write_covered_by_modifies => "writeCoveredByModifies",
        .read_preserved_by_frame => "readPreservedByFrame",
        .lock_covers_write => "lockCoversWrite",
        .external_call_frame => "externalCallFrame",
    };
}

fn assumptionKindName(kind: obligation.AssumptionKind) []const u8 {
    return switch (kind) {
        .requires => "requires",
        .assume => "assume",
        .path_assume => "pathAssume",
        .env_assume => "envAssume",
        .binding => "binding",
        .two_state_linkage => "twoStateLinkage",
        .frame => "frame",
        .loop_invariant => "loopInvariant",
        .callee_obligation => "calleeObligation",
        .callee_ensures => "calleeEnsures",
        .ghost_axiom => "ghostAxiom",
        .goal => "goal",
    };
}

test "Lean emitter writes locally nameless variable references" {
    const terms = [_]obligation.Term{
        .{ .variable = .{
            .free = .{
                .id = .{ .file_id = 0, .pattern_id = 7 },
                .name = "x",
                .ty = .{ .spelling = "u256" },
            },
        } },
        .{ .variable = .{
            .bound = .{
                .index = 0,
                .name = "i",
                .ty = .{ .spelling = "u256" },
            },
        } },
    };
    const set: obligation.ObligationSet = .{ .terms = &terms };

    var buffer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer buffer.deinit();
    try writeDataModule(&buffer.writer, set, .{ .namespace = "Ora.Generated.VarRefSmoke" });

    const rendered = buffer.written();
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".free { id := { file_id := 0, pattern_id := 7 }"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".bound { index := 0"));
}

test "Lean emitter writes effect frame obligations" {
    const declared_keys = [_]obligation.PlaceKey{.{ .parameter = 0 }};
    const declared = [_]obligation.PlaceRef{
        .{
            .root = "balances",
            .region = .storage,
            .keys = &declared_keys,
        },
    };
    const actual = declared;
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{},
            .phase = .ora_mlir,
            .origin = .source,
            .kind = .{ .effect_frame = .{
                .relation = .write_covered_by_modifies,
                .declared = &declared,
                .actual = &actual,
            } },
        },
    };
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{},
            .phase = .ora_mlir,
            .origin = .source,
            .backend = .lean,
            .kind = .obligation,
            .obligation_ids = &.{1},
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations, .queries = &queries };

    var buffer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer buffer.deinit();
    try writeModule(&buffer.writer, set, .{ .namespace = "Ora.Generated.EffectFrameSmoke" });

    const rendered = buffer.written();
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".effectFrame { relation := .writeCoveredByModifies"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "{ root := \"balances\", region := .storage"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "keys := [.parameter 0]"));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "def emittedQuery_2 : Prop :="));
    try std.testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "obligationFollowsFromAssumptions emittedManifest"));
}
